import os
import gym
import numpy as np
from gym import spaces
from simulators.simulator import Simulator
from utils.qoe import QoEModel
from utils.common import (allocate_tile_rates, generate_environment_samples, normalize_qoe_weight, 
                          normalize_size, normalize_quality, normalize_throughput, action2rates,
                          generate_environment_test_samples)


class SimpleRLEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, config, dataset, network_dataset, qoe_weights, log_path, startup_download, 
                 mode='train', seed=0, worker_num=1, device='cpu'):
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        self.config = config
        self.dataset = dataset
        self.network_dataset = network_dataset
        self.qoe_weights = qoe_weights
        self.log_path = log_path
        self.startup_download = startup_download
        self.video_rates = config.video_rates
        self.mode = mode
        self.random_seed = seed
        self.devide = device

        self.chunk_length = config.chunk_length
        self.past_k = config.past_k

        self.action_space = spaces.Discrete(n=len(self.video_rates))

        self.viewer = None
        
        self.videos = config.video_split[dataset][mode]
        self.users = config.user_split[dataset][mode]
        self.traces = config.network_split[network_dataset][mode]
        if mode != 'test':
            self.samples = generate_environment_samples(self.videos, self.users, self.traces, qoe_weights, seed=seed)
        else:
            self.samples = generate_environment_test_samples(self.videos, self.users, self.traces, qoe_weights)
        self.sample_id = -1
        self.sample_len = len(self.samples)

        self.worker_num = worker_num
        self.worker_id = seed % worker_num

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

        self.past_throughputs = None
        self.next_chunk_size = None
        self.rebuffer = None
        self.last_bitrates = None
        self.pred_viewport = None

        self.gt_viewport = None

        # for log
        self.log_qoe = []
        self.log_qoe1 = []
        self.log_qoe2 = []
        self.log_qoe3 = []
        
    def reset(self, seed=None, options=None):
        self.sample_id = self.worker_id
        self.worker_id = (self.worker_id + self.worker_num) % self.sample_len

        self.current_video = self.videos[self.samples[self.sample_id][0]]
        self.current_user = self.users[self.samples[self.sample_id][1]]
        self.current_trace = self.traces[self.samples[self.sample_id][2]]
        self.current_qoe_weight = np.array(self.qoe_weights[self.samples[self.sample_id][3]], dtype=np.float32)

        self.simulator = Simulator(config=self.config,
                                   dataset=self.dataset,
                                   video=self.current_video,
                                   user=self.current_user,
                                   network_dataset=self.network_dataset,
                                   trace=self.current_trace,
                                   startup_download=self.startup_download)
        self.qoe_model = QoEModel(self.config, *self.current_qoe_weight)

        self.tile_rates = []

        self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()
        self.next_chunk_size = self.simulator.get_next_chunk_size()

        self.last_bitrates = np.zeros(2, dtype=np.float32)
        self.rebuffer = np.zeros(1, dtype=np.float32) 
        self.past_throughputs = np.zeros((1, self.past_k), dtype=np.float32)

        self.state = {
            'throughput': self.past_throughputs,
            'chunk_sizes': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
            'rebuffer': self.rebuffer,
            'last_bitrates': self.last_bitrates,
            'pred_viewport': self.pred_viewport,
        }

        return self.state

    def step(self, action):
        rate_in, rate_out = action2rates(action)
        self.tile_rates, _ = allocate_tile_rates(rate_version_in=rate_in, rate_version_out=rate_out, pred_viewport=self.pred_viewport,
                                                 video_rates=self.video_rates, tile_num_width=self.tile_num_width, tile_num_height=self.tile_num_height)
        self.tile_rates = list(self.tile_rates)
                
        selected_tile_sizes, selected_tile_quality, chunk_size, chunk_quality, download_time, \
        rebuffer_time, actual_viewport, over = self.simulator.simulate_download(self.tile_rates)
        qoe, qoe1, qoe2, qoe3 = self.qoe_model.calculate_qoe(actual_viewport=self.gt_viewport,
                                                             tile_quality=selected_tile_quality,
                                                             rebuffer_time=rebuffer_time,)
        if self.mode != 'train':
            reward = qoe
        else:
            reward = qoe / sum(self.current_qoe_weight)
        self.log_qoe.append(float(qoe))
        self.log_qoe1.append(float(qoe1))
        self.log_qoe2.append(float(qoe2))
        self.log_qoe3.append(float(qoe3))

        self.past_throughputs = np.roll(self.past_throughputs, 1)
        self.past_throughputs[0, 0] = normalize_throughput(self.config, chunk_size / download_time)
        self.rebuffer[0] = qoe2
        self.last_bitrates[0], self.last_bitrates[1] = self.video_rates[rate_in], self.video_rates[rate_out]
        self.last_bitrates = normalize_quality(self.config, self.last_bitrates)
        
        if over:
            self.state.update({
                'throughput': self.past_throughputs,
                'chunk_sizes': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
                'rebuffer': self.rebuffer,
                'last_bitrates': self.last_bitrates,
                'pred_viewport': self.pred_viewport,
            })
            self._log()
        else:
            self.next_chunk_size = self.simulator.get_next_chunk_size()
            self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()

            self.state.update({
                'throughput': self.past_throughputs,
                'chunk_sizes': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
                'rebuffer': self.rebuffer,
                'last_bitrates': self.last_bitrates,
                'pred_viewport': self.pred_viewport,
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
