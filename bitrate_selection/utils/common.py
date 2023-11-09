import torch
import numpy as np
import yaml
import math
from munch import Munch
from collections import deque
from prettytable import PrettyTable


DEFAULT_CONFIG_YML_PATH = '../config.yml'


def get_config_from_yml(config_yml_path=None):
    if config_yml_path is None:
        config_yml_path = DEFAULT_CONFIG_YML_PATH
    
    with open(config_yml_path, 'r', encoding='utf8') as config_yml_file:
        config = yaml.load(config_yml_file, Loader=yaml.SafeLoader)
        config_yml_file.close()
        
    config = Munch(config)
    # concat datasets_base_dir with other datasets_dir
    datasets_dir_list = [config.raw_datasets_dir, config.raw_network_datasets_dir, config.viewport_datasets_dir,
                         config.video_datasets_dir, config.network_datasets_dir]
    for datasets_dir in datasets_dir_list:
        for key in datasets_dir.keys():
            datasets_dir[key] =  config.datasets_base_dir + datasets_dir[key]

    # concat results_base_dir with other results_dir
    config.vp_results_dir = config.results_base_dir + config.vp_results_dir
    config.bs_results_dir = config.results_base_dir + config.bs_results_dir

    # concat models_base_dir with other models_dir
    config.vp_models_dir = config.models_base_dir + config.vp_models_dir
    config.bs_models_dir = config.models_base_dir + config.bs_models_dir

    return config


def normalize_quality(config, quality):
    max_quality = config.video_rates[-1]  # since we use bitrate as quality, the max quality is the max bitrate
    return quality / max_quality


def normalize_size(config, size):
    max_size = config.max_size
    return size / max_size


def normalize_throughput(config, throughput):
    max_throughput = config.max_throughput
    return throughput / max_throughput


def normalize_qoe_weight(qoe_weight):
    weight_sum = sum(qoe_weight)
    return qoe_weight / weight_sum


def generate_environment_samples(video_list, user_list, trace_list, qoe_list, seed=0):
    """
    Exhaustive training over all possible environments is expensive and unnecessary.
    So this function samples some environments while making sure that each video/user/trace/qoe will be used at least once 
    """
    def _sample_ids(_id_list, _list_len, _max_len):
        _ret = []
        for _i in range(_max_len):
            _ret.append(_id_list[_i % _list_len])
        return _ret

    video_list_len = len(video_list)
    user_list_len = len(user_list)
    trace_list_len = len(trace_list)
    qoe_list_len = len(qoe_list)
    max_len = max(video_list_len, user_list_len, trace_list_len, qoe_list_len)
    total_len = max(max_len, video_list_len * qoe_list_len * math.ceil(max_len / (video_list_len * qoe_list_len)))

    video_ids = _sample_ids(list(range(video_list_len)), video_list_len, total_len)
    user_ids = _sample_ids(list(range(user_list_len)), user_list_len, total_len)
    trace_ids = _sample_ids(list(range(trace_list_len)), trace_list_len, total_len)
    qoe_ids = _sample_ids(list(range(qoe_list_len)), qoe_list_len, total_len)

    environment_samples = list(zip(video_ids, user_ids, trace_ids, qoe_ids))
    return environment_samples


def generate_environment_test_samples(video_list, user_list, trace_list, qoe_list):
    """
    Unlike training, testing on all possible environments is necessary.
    """
    video_list_len = len(video_list)
    user_list_len = len(user_list)
    trace_list_len = len(trace_list)
    qoe_list_len = len(qoe_list)

    environment_samples = [(i, j, k, l) for i in range(video_list_len) for j in range(user_list_len)
                           for k in range(trace_list_len) for l in range(qoe_list_len)]
    return environment_samples


def action2rates(action):
    # tansform discrete 10 to rate in and rate out
    rate_in, rate_out = 0, 0
    if action == 0: rate_in, rate_out = 1, 0
    if action == 1: rate_in, rate_out = 2, 0
    if action == 2: rate_in, rate_out = 3, 0
    if action == 3: rate_in, rate_out = 4, 0
    if action == 4: rate_in, rate_out = 2, 1
    if action == 5: rate_in, rate_out = 3, 1
    if action == 6: rate_in, rate_out = 4, 1
    if action == 7: rate_in, rate_out = 3, 2
    if action == 8: rate_in, rate_out = 4, 2
    if action == 9: rate_in, rate_out = 4, 3
    if action == 10: rate_in, rate_out = 0, 0
    if action == 11: rate_in, rate_out = 1, 1
    if action == 12: rate_in, rate_out = 2, 2
    if action == 13: rate_in, rate_out = 3, 3
    if action == 14: rate_in, rate_out = 4, 4
    return rate_in, rate_out


def rates2action(rate_in, rate_out):
    action = 0
    if rate_in == 1 and rate_out == 0: action = 0
    if rate_in == 2 and rate_out == 0: action = 1
    if rate_in == 3 and rate_out == 0: action = 2
    if rate_in == 4 and rate_out == 0: action = 3
    if rate_in == 2 and rate_out == 1: action = 4
    if rate_in == 3 and rate_out == 1: action = 5
    if rate_in == 4 and rate_out == 1: action = 6
    if rate_in == 3 and rate_out == 2: action = 7
    if rate_in == 4 and rate_out == 2: action = 8
    if rate_in == 4 and rate_out == 3: action = 9
    if rate_in == 0 and rate_out == 0: action = 10
    if rate_in == 1 and rate_out == 1: action = 11
    if rate_in == 2 and rate_out == 2: action = 12
    if rate_in == 3 and rate_out == 3: action = 13
    if rate_in == 4 and rate_out == 4: action = 14
    return action


def allocate_tile_rates(rate_version_in: int, rate_version_out: int, pred_viewport: np.ndarray,
                        video_rates, tile_num_width, tile_num_height):
    """
    Allocate bitrates to tiles according to our pyramid allocation strategy.
    rate_version_in: the bitrate version for tiles inside the viewport
    rate_version_out: the bitrate versionh for tiles outside the viewport
    pred_viewport: predicted viewport
    video_rates: all video bitrates
    """
    pred_viewport = pred_viewport.reshape(tile_num_height, tile_num_width)
    insides_viewports = list(zip(*np.where(pred_viewport == 1)))

    scales = np.zeros(pred_viewport.shape, dtype=np.int32)
    visited = np.zeros(pred_viewport.shape, dtype=np.bool_)
    for tile in insides_viewports:
        visited[tile] = np.True_
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1)]
    tile_queue = deque(insides_viewports)
    while tile_queue:
        current_tile = tile_queue.popleft()
        for direction in directions:
            next_tile = (current_tile[0] + direction[0]) % tile_num_height, \
                        (current_tile[1] + direction[1]) % tile_num_width
            if not visited[next_tile]:
                scales[next_tile] = scales[current_tile] + 1
                tile_queue.append(next_tile)
                visited[next_tile] = np.True_

    def find_closest_rate_version(_all_rates, _rate):
        _rate_version = 0
        _gap = abs(_all_rates[_rate_version] - _rate)
        for _i in range(len(_all_rates)):
            _tmp_gap = abs(_all_rates[_i] - _rate)
            if _tmp_gap < _gap:
                _rate_version = _i
                _gap = _tmp_gap
            elif _tmp_gap == _gap and _all_rates[_i] < _all_rates[_rate_version]:
                _rate_version = _i
        return _rate_version

    tile_rate_versions = np.zeros(pred_viewport.shape, dtype=np.int32)
    tile_rates = np.zeros(pred_viewport.shape, dtype=np.int32)
    tile_rate_versions[scales == 0] = rate_version_in
    tile_rates[scales == 0] = video_rates[rate_version_in]
    max_scale = np.max(scales)
    for scale in range(1, max_scale + 1):
        rate = find_closest_rate_version(video_rates, video_rates[rate_version_out] // scale)
        tile_rate_versions[scales == scale] = rate
        tile_rates[scales == scale] = video_rates[rate]
    # print(tile_rate_versions)
    # print(tile_rates)
    return tile_rate_versions.reshape(-1), tile_rates.reshape(-1)


def read_log_file(log_path):
    pt = PrettyTable()
    pt.field_names = ['video', 'user', 'trace', 'qoe_w1', 'qoe_w2', 'qoe_3', 'qoe', 'qoe1', 'qoe2', 'qoe3']
    mean_qoe, mean_qoe1, mean_qoe2, mean_qoe3 = 0., 0., 0., 0.
    sample_count = 0
    with open(log_path, 'r') as file:
        file.readline()
        for line in file.readlines():
            line = line.strip().split(',')
            video, user, trace = list(map(int, line[:3]))
            qoe_w1, qoe_w2, qoe_w3, qoe, qoe1, qoe2, qoe3 = list(map(float, line[3:]))
            mean_qoe += qoe
            mean_qoe1 += qoe1
            mean_qoe2 += qoe2
            mean_qoe3 += qoe3
            pt.add_row([video, user, trace, qoe_w1, qoe_w2, qoe_w3, qoe, qoe1, qoe2, qoe3])
            sample_count += 1
    mean_qoe /= sample_count
    mean_qoe1 /= sample_count
    mean_qoe2 /= sample_count
    mean_qoe3 /= sample_count
    pt.add_row([-1, -1, -1, -1, -1, -1, mean_qoe, mean_qoe1, mean_qoe2, mean_qoe3])
    print(pt)
