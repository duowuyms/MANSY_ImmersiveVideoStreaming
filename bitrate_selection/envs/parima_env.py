import os
import gym
import pickle
import numpy as np
# import sys
# sys.path.append('/home/wuduo/notmuch/projects/2023_omnidirectional_vs/codes/bitrate_selection')
from tqdm import trange
from simulators.simulator import Simulator
from utils.common import (normalize_quality, normalize_size, normalize_throughput, normalize_qoe_weight, 
                          action2rates, rates2action, allocate_tile_rates)
from utils.qoe import QoEModel


class ParimaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, config, dataset, network_dataset, qoe_weights, samples, log_path, startup_download, 
                 dataset_frequency=5, mode='train', seed=0, device='cpu'):
        self.config = config
        self.dataset = dataset
        self.network_dataset = network_dataset
        self.qoe_weights = qoe_weights
        self.samples = samples
        self.log_path = log_path
        self.startup_download = startup_download
        self.video_rates = config.video_rates
        self.dataset_frequency = dataset_frequency
        self.mode = mode
        self.random_seed = seed
        self.device = device

        self.chunk_length = config.chunk_length
        self.past_k = config.past_k
    
        self.videos = config.video_split[dataset][mode]
        self.users = config.user_split[dataset][mode]
        self.traces = config.network_split[network_dataset][mode]
        self.sample_id = -1
        self.sample_len = len(self.samples)

        self.current_video = None
        self.current_user = None
        self.current_trace = None
        self.current_qoe_weight = None
        self.simulator = None
        self.qoe_model = None

        self.tile_num = config.tile_total_num
        self.tile_num_width = config.tile_num_width
        self.tile_num_height = config.tile_num_height
        self.tile_rates = []

        self.gt_viewport = None
        self.pred_viewport = None
        self.past_pred_accuracy = None
        self.next_chunk_size = None
        self.next_chunk_quality = None

        self.past_throughputs = None
        self.viewport_centers = None
        self.vp_center_start_frame = None

        self.state = None
        self.device = device

        # for log
        self.log_qoe = []
        self.log_qoe1 = []
        self.log_qoe2 = []
        self.log_qoe3 = []
    
    def reset(self):
        self.sample_id = (self.sample_id + 1) % self.sample_len
        self.current_video = self.videos[self.samples[self.sample_id][0]]
        self.current_user = self.users[self.samples[self.sample_id][1]]
        self.current_trace = self.traces[self.samples[self.sample_id][2]]
        self.current_qoe_weight = np.array(self.qoe_weights[self.samples[self.sample_id][3]], dtype=np.float32)
        # print(self.mode, self.sample_id, self.sample_len, self.current_video, self.current_user, self.current_trace, self.current_qoe_weight)

        vp_center_path = os.path.join(self.config.viewport_datasets_dir[self.dataset], 'prediction',
                                      f'video{self.current_video}',  f'user{self.current_user}_vp_center.pkl')
        self.viewport_centers = pickle.load(open(vp_center_path, 'rb'))
        self.vp_center_start_frame = self.viewport_centers[0][0]

        self.simulator = Simulator(config=self.config,
                                   dataset=self.dataset,
                                   network_dataset=self.network_dataset,
                                   video=self.current_video,
                                   user=self.current_user,
                                   trace=self.current_trace,
                                   startup_download=self.startup_download)
        self.qoe_model = QoEModel(self.config, *self.current_qoe_weight)

        self.tile_rates = []

        self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()

        self.next_chunk_size = self.simulator.get_next_chunk_size()
        self.next_chunk_quality = self.simulator.get_next_chunk_quality()

        self.past_throughputs = np.zeros((1, self.past_k), dtype=np.float32)

        self.state = {
            'throughput': self.past_throughputs,
        }

        return self.state

    def step(self, rates):
        self.tile_rates = list(rates)
        selected_tile_sizes, selected_tile_quality, chunk_size, chunk_quality, download_time, \
        rebuffer_time, actual_viewport, over = self.simulator.simulate_download(self.tile_rates)
        qoe, qoe1, qoe2, qoe3 = self.qoe_model.calculate_qoe(actual_viewport=self.gt_viewport,
                                                             tile_quality=selected_tile_quality,
                                                             rebuffer_time=rebuffer_time,)
        
        reward = qoe
        self.log_qoe.append(float(qoe))
        self.log_qoe1.append(float(qoe1))
        self.log_qoe2.append(float(qoe2))
        self.log_qoe3.append(float(qoe3))
        
        self.past_throughputs = np.roll(self.past_throughputs, 1)
        self.past_throughputs[0, 0] = chunk_size / download_time

        if over:
            self.state.update({
                'throughput': self.past_throughputs,
            })
            # print(self.simulator.start_chunk, self.simulator.end_chunk, self.simulator.next_chunk)
            self._log()
        else:
            self.next_chunk_size = self.simulator.get_next_chunk_size()
            self.next_chunk_quality = self.simulator.get_next_chunk_quality()
            self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()

            self.state.update({
                'throughput': self.past_throughputs,
            })
        
        return self.state, reward, over, {}

    def sample_count(self):
        return len(self.samples)

    def seed(self, seed):
        np.random.seed(seed)
        self.seed = seed
        self.worker_id = seed % self.worker_num
        
    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _log(self):
        if not os.path.exists(self.log_path):
            heads = 'video,user,trace,qoe_w1,qoe_w2,qoe_w3,qoe,qoe1,qoe2,qoe3\n'
            with open(self.log_path, 'w', encoding='utf-8') as file:
                file.write(heads)
                file.close()
        with open(self.log_path, 'a', encoding='utf-8') as file:
            qoe_w1, qoe_w2, qoe_w3 = self.current_qoe_weight
            qoe = round(sum(self.log_qoe) / len(self.log_qoe) / sum(self.current_qoe_weight), 5)  # normalized qoe
            qoe1 = round(sum(self.log_qoe1) / len(self.log_qoe1), 5)
            qoe2 = round(sum(self.log_qoe2) / len(self.log_qoe2), 5)
            qoe3 = round(sum(self.log_qoe3) / len(self.log_qoe3), 5)
            line = f'{self.current_video},{self.current_user},{self.current_trace},{qoe_w1},{qoe_w2},{qoe_w3},' \
                   f'{qoe},{qoe1},{qoe2},{qoe3}\n'
            file.write(line)
            file.close()
        self.log_qoe.clear()
        self.log_qoe1.clear()
        self.log_qoe2.clear()
        self.log_qoe3.clear() 
    
    def estimate_bandwidth(self, aggressive=1.0):
        num, harmonic_mean = 0, 0.
        for throughput in self.past_throughputs.reshape(-1).tolist():
            if throughput != 0:
                harmonic_mean += 1 / throughput
                num += 1
        bandwidth = 0.
        if num != 0:
            bandwidth = num / harmonic_mean
        return aggressive * bandwidth

    def choose_action(self, bandwidth):
        current_chunk = self.simulator.next_chunk
        weights = np.ones((self.tile_num_height, self.tile_num_width), dtype=np.float32)
        pred_viewport = self.pred_viewport.reshape(self.tile_num_height, self.tile_num_width)
        for frame_id in range(self.dataset_frequency):
            frame = current_chunk * self.dataset_frequency + frame_id
            actual_frame, vp_center_x, vp_center_y = self.viewport_centers[frame - self.vp_center_start_frame]
            assert frame == actual_frame
            dis1 = vp_center_x + vp_center_y
            dis2 = self.tile_num_width - vp_center_x - 1 + vp_center_y
            dis3 = vp_center_x + self.tile_num_height - vp_center_y - 1
            dis4 = self.tile_num_width - vp_center_x - 1 + self.tile_num_height - vp_center_y - 1
            max_dis = max(dis1, dis2, dis3, dis4)

            weights[vp_center_x, vp_center_y] += 1
            for x in range(self.tile_num_width):
                for y in range(self.tile_num_height):
                    if x == vp_center_x and y == vp_center_y:
                        continue
                    dis = abs(vp_center_x - x) + abs(vp_center_y - y) 
                    if pred_viewport[y, x] == 1:
                        weights[x, y] += 1 - dis / (2 * max_dis)
                    else:
                        weights[x, y] += 1 - dis / max_dis
        bandwidth_each_tile = weights / np.sum(weights) * bandwidth
        bandwidth_each_tile = bandwidth_each_tile.reshape(-1)
        tile_rates = np.zeros(self.tile_num, dtype=np.int32)
        for i in range(self.tile_num):
            nearest_gap = float('inf')
            nearest_rate = 0
            for rate in range(len(self.video_rates)):
                gap = abs(self.next_chunk_size[rate][i] - bandwidth_each_tile[i])
                if gap < nearest_gap:
                    nearest_gap = gap
                    nearest_rate = rate
            tile_rates[i] = nearest_rate
        return tile_rates

