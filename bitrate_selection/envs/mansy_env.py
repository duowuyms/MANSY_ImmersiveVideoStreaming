import os
import gym
import numpy as np
from simulators.simulator import Simulator
from gym import spaces
from utils.common import (normalize_quality, normalize_size, normalize_throughput, normalize_qoe_weight, 
                          action2rates, rates2action, allocate_tile_rates, generate_environment_samples,
                          generate_environment_test_samples)
from utils.mansy_utils import calculate_indentifier_reward
from utils.qoe import QoEModel


CNT = 0
OBSERVE_ROUND = 500

class MANSYEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, config, dataset, network_dataset, qoe_weights, identifier, lamb, log_path, 
                 startup_download, mode='train', seed=0, worker_num=1, device='cpu', use_identifier=False):
        super().__init__()
        assert mode in ['train', 'valid', 'test']
        self.config = config
        self.identifier = identifier
        self.lamb = lamb
        self.dataset = dataset
        self.network_dataset = network_dataset
        self.qoe_weights = qoe_weights
        self.log_path = log_path
        self.startup_download = startup_download
        self.video_rates = config.video_rates
        self.mode = mode
        self.random_seed = seed
        self.devide = device
        self.use_identifier = use_identifier
        print('Use Identifier:', use_identifier)

        self.chunk_length = config.chunk_length
        self.past_k = config.past_k

        self.action_space = spaces.Discrete(n=config.action_space)

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

        self.gt_viewport = None
        self.pred_viewport = None
        self.previous_pred_accuracy = None
        self.last_chunk_accuracy = 0

        self.next_chunk_size = None
        self.next_chunk_quality = None

        self.previous_tiles_quality = None
        self.previous_tiles_prob = None
        self.previous_tiles_total_size = None

        self.last_chunk_quality = None
        self.buffer = None
        self.past_throughputs = None
        self.past_bitrates_inside_viewports = None
        self.past_bitrates_outside_viewports = None
        self.past_viewport_qualities = None
        self.past_quality_variances = None
        self.past_rebuffering = None
        self.state = None
        self.device = device

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
        self.past_pred_accuracy = np.zeros((1, self.past_k), dtype=np.float32)
        self.last_chunk_accuracy = accuracy

        self.next_chunk_size = self.simulator.get_next_chunk_size()
        self.next_chunk_quality = self.simulator.get_next_chunk_quality()

        self.last_chunk_quality = np.zeros(1, dtype=np.float32)
        self.buffer = np.zeros(1, dtype=np.float32)
        self.buffer[0] = self.simulator.get_buffer_size()
        self.past_throughputs = np.zeros((1, self.past_k), dtype=np.float32)
        self.past_bitrates_inside_viewports = np.zeros((1, self.past_k), dtype=np.float32)
        self.past_bitrates_outside_viewports = np.zeros((1, self.past_k), dtype=np.float32)
        self.past_viewport_qualities = np.zeros((1, self.past_k), dtype=np.float32)
        self.past_quality_variances = np.zeros((1, self.past_k), dtype=np.float32)
        self.past_rebuffering = np.zeros((1, self.past_k), dtype=np.float32)

        self.state = {
            'throughput': self.past_throughputs,
            'next_chunk_size': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
            'next_chunk_quality': normalize_quality(self.config, self.next_chunk_quality).astype(np.float32),
            'pred_viewport': self.pred_viewport.reshape(1, -1),
            'rates_inside': self.past_bitrates_inside_viewports,
            'rates_outside': self.past_bitrates_outside_viewports,
            'viewport_acc': self.past_pred_accuracy,
            'buffer': self.buffer / self.startup_download,
            'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
            'action_one_hot': np.zeros(self.config.action_space, dtype=np.float32),
            'past_viewport_qualities': self.past_viewport_qualities,
            'past_quality_variances': self.past_quality_variances,
            'past_rebuffering': self.past_rebuffering
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
        
        action_one_hot = np.zeros(self.config.action_space, dtype=np.float32)
        action_one_hot[action] = 1.
        if self.mode != 'train' or not self.use_identifier:
            reward = qoe
        else:
            # identifier_reward = calculate_indentifier_reward(self.identifier, state=self.state, action_one_hot=action_one_hot)
            # reward = (1 - self.lamb) * qoe / sum(self.current_qoe_weight) + self.lamb * identifier_reward
            # global CNT, OBSERVE_ROUND
            # CNT = CNT + 1
            # if CNT % OBSERVE_ROUND == 0:
            #     print('Reward:', reward, ' --- ', 'QoE Reward:', qoe / sum(self.current_qoe_weight), ' --- ', 'Identifier Reward:', identifier_reward)
            reward = qoe / sum(self.current_qoe_weight)
            # print(qoe / sum(self.current_qoe_weight), identifier_reward, reward)
        # if self.mode == 'train':
        #     reward = qoe / sum(self.current_qoe_weight)
        # elif self.mode == 'valid':
        #     identifier_reward = calculate_indentifier_reward(self.identifier, state=self.state, action_one_hot=action_one_hot)
        #     reward = (1 - self.lamb) * qoe / sum(self.current_qoe_weight) + self.lamb * identifier_reward
        # else:
        #     reward = qoe
            
        self.log_qoe.append(float(qoe))
        self.log_qoe1.append(float(qoe1))
        self.log_qoe2.append(float(qoe2))
        self.log_qoe3.append(float(qoe3))

        self.past_throughputs = np.roll(self.past_throughputs, 1)
        self.past_throughputs[0, 0] = normalize_throughput(self.config, chunk_size / download_time)
        self.past_pred_accuracy = np.roll(self.past_pred_accuracy, 1)
        self.past_pred_accuracy[0, 0] = self.last_chunk_accuracy
        self.past_bitrates_inside_viewports = np.roll(self.past_bitrates_inside_viewports, 1)
        self.past_bitrates_inside_viewports[0, 0] = normalize_quality(self.config, self.video_rates[rate_in])
        self.past_bitrates_outside_viewports = np.roll(self.past_bitrates_outside_viewports, 1)
        self.past_bitrates_outside_viewports[0, 0] = normalize_quality(self.config, self.video_rates[rate_out])
        self.buffer[0] = self.simulator.get_buffer_size()
        self.past_viewport_qualities = np.roll(self.past_viewport_qualities, 1)
        self.past_viewport_qualities[0, 0] = qoe1
        self.past_rebuffering = np.roll(self.past_rebuffering, 1)
        self.past_rebuffering[0, 0] = qoe2 / self.startup_download
        self.past_quality_variances = np.roll(self.past_quality_variances, 1)
        self.past_quality_variances[0, 0] = qoe3

        if over:
            self.state.update({
                'throughput': self.past_throughputs,
                'next_chunk_size': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
                'next_chunk_quality': normalize_quality(self.config, self.next_chunk_quality).astype(np.float32),
                'pred_viewport': self.pred_viewport.reshape(1, -1),
                'rates_inside': self.past_bitrates_inside_viewports,
                'rates_outside': self.past_bitrates_outside_viewports,
                'viewport_acc': self.past_pred_accuracy,
                'buffer': self.buffer / self.startup_download,
                'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
                'action_one_hot': action_one_hot,
                'past_viewport_qualities': self.past_viewport_qualities,
                'past_quality_variances': self.past_quality_variances,
                'past_rebuffering': self.past_rebuffering,
            })
            # print(self.simulator.start_chunk, self.simulator.end_chunk, self.simulator.next_chunk)
            self._log()
        else:
            self.next_chunk_size = self.simulator.get_next_chunk_size()
            self.next_chunk_quality = self.simulator.get_next_chunk_quality()
            self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()
            self.last_chunk_accuracy = accuracy

            self.state.update({
                'throughput': self.past_throughputs,
                'next_chunk_size': normalize_size(self.config, self.next_chunk_size).astype(np.float32),
                'next_chunk_quality': normalize_quality(self.config, self.next_chunk_quality).astype(np.float32),
                'pred_viewport': self.pred_viewport.reshape(1, -1),
                'rates_inside': self.past_bitrates_inside_viewports,
                'rates_outside': self.past_bitrates_outside_viewports,
                'viewport_acc': self.past_pred_accuracy,
                'buffer': self.buffer / self.startup_download,
                'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
                'action_one_hot': action_one_hot,
                'past_viewport_qualities': self.past_viewport_qualities,
                'past_quality_variances': self.past_quality_variances,
                'past_rebuffering': self.past_rebuffering,
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
