import torch
import numpy as np
import yaml
from munch import Munch


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


def find_block_covered_by_point(x: int, y: int, block_width: int, block_height: int):
    w, h = x // block_width, y // block_height
    if x > 0 and x % block_width == 0:
        w -= 1
    if y > 0 and y % block_height == 0:
        h -= 1
    return w, h


def find_tiles_covered_by_viewport(x, y, video_width, video_height, tile_width, tile_height, 
                                   tile_num_width, tile_num_height, fov_width=600, fov_height=300):
    viewport = np.zeros([tile_num_height, tile_num_width], dtype=np.uint8)
    half_fov_width = fov_width // 2
    half_fov_height = fov_height // 2
    fov_x1, fov_y1 = x - half_fov_width, y - half_fov_height
    fov_x2, fov_y2 = x + half_fov_width, y + half_fov_height
    regions = _find_regions_covered_by_fov(fov_x1, fov_y1, fov_x2, fov_y2, video_width, video_height)
    for region_x1, region_y1, region_x2, region_y2 in regions:
        tile_x1, tile_y1 = find_block_covered_by_point(region_x1, region_y1, tile_width, tile_height)
        tile_x2, tile_y2 = find_block_covered_by_point(region_x2, region_y2, tile_width, tile_height)
        viewport[tile_y1:tile_y2 + 1, tile_x1:tile_x2 + 1] = 1
    return viewport


def to_position_normalized_cartesian(values):
    """
    Bound position within [0, 1]^2.
    """
    values_clone = values.clone()
    indices_negative = values < 0
    indices_greater_than_one = values > 1
    values_clone[indices_negative] = values[indices_negative] - values[indices_negative].to(dtype=torch.int) + 1
    values_clone[indices_greater_than_one] = values[indices_greater_than_one] - values[indices_greater_than_one].to(dtype=torch.int)
    return values_clone


def mean_square_error(position_a, position_b, dimension=2):
    """
    Mean square error that considers the periodicity of viewport positions.
    """
    error = torch.abs(position_a - position_b)
    error = torch.minimum(error, torch.abs(position_a + 1 - position_b))
    error = torch.minimum(error, torch.abs(position_a - 1 - position_b))
    return torch.sum(error * error, dim=-1) / dimension


def _find_regions_covered_by_fov(fov_x1, fov_y1, fov_x2, fov_y2, video_width, video_height):
    if fov_x1 >= 0 and fov_x2 <= video_width and fov_y1 >= 0 and fov_y2 <= video_height:
        regions = [(fov_x1, fov_y1, fov_x2, fov_y2)]
    else:
        # case 2
        if fov_x1 < 0 and fov_x2 <= video_width and fov_y1 < 0 and fov_y2 <= video_height:
            regions = [(0, 0, fov_x2, fov_y2),
                       (fov_x1 % video_width, 0, video_width, fov_y2),
                       (0, fov_y1 % video_height, fov_x2, video_height),
                       (fov_x1 % video_width, fov_y1 % video_height, video_width, video_height)]
        # case 3
        elif fov_x1 >= 0 and fov_x2 > video_width and fov_y1 < 0 and fov_y2 <= video_height:
            regions = [(0, 0, fov_x2 % video_width, fov_y2),
                       (fov_x1, 0, video_width, fov_y2),
                       (0, fov_y1 % video_height, fov_x2 % video_width, video_height),
                       (fov_x1, fov_y1 % video_height, video_width, video_height)]
        # case 4
        elif fov_x1 < 0 and fov_x2 <= video_width and fov_y1 >= 0 and fov_y2 > video_height:
            regions = [(0, 0, fov_x2, fov_y2 % video_height),
                       (fov_x1 % video_width, 0, video_width, fov_y2 % video_height),
                       (0, fov_y1, fov_x2, video_height),
                       (fov_x1 % video_width, fov_y1, video_width, video_height)]
        # case 5
        elif fov_x1 >= 0 and fov_x2 > video_width and fov_y1 >= 0 and fov_y2 > video_height:
            regions = [(0, 0, fov_x2 % video_width, fov_y2 % video_height),
                       (fov_x1, 0, video_width, fov_y2 % video_height),
                       (0, fov_y1, fov_x2 % video_width, video_height),
                       (fov_x1, fov_y1, video_width, video_height)]
        # case 6
        elif fov_x1 < 0 and fov_x2 <= video_width and fov_y1 >= 0 and fov_y2 <= video_height:
            regions = [(0, fov_y1, fov_x2, fov_y2),
                       (fov_x1 % video_width, fov_y1, video_width, fov_y2)]
        # case 7
        elif fov_x1 >= 0 and fov_x2 > video_width and fov_y1 >= 0 and fov_y2 <= video_height:
            regions = [(0, fov_y1, fov_x2 % video_width, fov_y2),
                       (fov_x1, fov_y1, video_width, fov_y2)]
        # case 8
        elif fov_x1 >= 0 and fov_x2 <= video_width and fov_y1 < 0 and fov_y2 <= video_height:
            regions = [(fov_x1, 0, fov_x2, fov_y2),
                       (fov_x1, fov_y1 % video_height, fov_x2, video_height)]
        # case 9
        elif fov_x1 >= 0 and fov_x2 <= video_width and fov_y1 >= 0 and fov_y2 > video_height:
            regions = [(fov_x1, 0, fov_x2, fov_y2 % video_height),
                       (fov_x1, fov_y1, fov_x2, video_height)]
    return regions
