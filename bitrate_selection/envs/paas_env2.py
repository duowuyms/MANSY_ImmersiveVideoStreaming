# preference-aware action-value learning with dynamic-preference training

import os
import random
import gym
import numpy as np
# import sys
# sys.path.append('/home/wuduo/notmuch/projects/2023_omnidirectional_vs/codes/bitrate_selection')
from random import choice
from gym import spaces
from simulators.simulator import Simulator
from utils.qoe import QoEModelPAAS
from utils.common import (allocate_tile_rates, generate_environment_samples, normalize_qoe_weight, 
                          normalize_quality, normalize_throughput, action2rates, generate_environment_test_samples)


class PAASEnv(gym.Env):
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

        self.gt_viewport = None
        self.pred_viewport = None
        self.previous_pred_accuracy = None
        self.last_chunk_accuracy = 0

        self.next_chunk_size = None
        self.next_chunk_quality = None

        self.last_viewport_quality = None
        self.buffer = None
        self.past_throughputs = None

        # for log
        self.log_qoe = []
        self.log_qoe1 = []
        self.log_qoe2 = []
        self.log_qoe3 = []

    def reset(self, seed=None, options=None):
        # if self.mode == 'train':
        #     self.sample_id = random.randint(0, self.sample_len - 1)
        # else:
        #     self.sample_id = (self.seed + 1) % self.sample_len
        #     self.seed += 1
        #     print(self.mode, self.sample_id, self.sample_len)
        self.sample_id = self.worker_id
        self.worker_id = (self.worker_id + self.worker_num) % self.sample_len

        self.current_video = self.videos[self.samples[self.sample_id][0]]
        self.current_user = self.users[self.samples[self.sample_id][1]]
        self.current_trace = self.traces[self.samples[self.sample_id][2]]
        self.current_qoe_weight = np.array(self.qoe_weights[self.samples[self.sample_id][3]], dtype=np.float32)
        # print(self.mode, self.sample_id, self.sample_len, self.current_video, self.current_user, self.current_trace, self.current_qoe_weight)

        self.simulator = Simulator(config=self.config,
                                   dataset=self.dataset,
                                   video=self.current_video,
                                   user=self.current_user,
                                   network_dataset=self.network_dataset,
                                   trace=self.current_trace,
                                   startup_download=self.startup_download)
        self.qoe_model = QoEModelPAAS(self.config, *self.current_qoe_weight)

        self.tile_rates = []

        self.gt_viewport, self.pred_viewport, accuracy = self.simulator.get_viewport()
        self.previous_pred_accuracy = np.zeros((1, self.past_k), dtype=np.float32)
        self.last_chunk_accuracy = accuracy

        self.next_chunk_size = self.simulator.get_next_chunk_size()
        self.next_chunk_quality = self.simulator.get_next_chunk_quality()

        self.last_viewport_quality = np.zeros(1, dtype=np.float32)
        self.buffer = np.zeros(1, dtype=np.float32) 
        self.buffer[0] = self.simulator.get_buffer_size()
        self.past_throughputs = np.zeros((1, self.past_k), dtype=np.float32)

        # state in PAAS
        self.state = {
            'throughput': self.past_throughputs,
            'buffer': self.buffer,
            'last_viewport_quality': self.last_viewport_quality,
            'chunk_bitrates': self.next_chunk_quality,
            'pred_viewport': self.pred_viewport,
            'viewport_acc': self.previous_pred_accuracy,
            'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
            'sample_weight': normalize_qoe_weight(self.current_qoe_weight),
            'sample_weight_qoe': 0.
        }

        return self.state

    def step(self, action):
        rate_in, rate_out = action2rates(action)
        self.tile_rates, _ = allocate_tile_rates(rate_version_in=rate_in, rate_version_out=rate_out, pred_viewport=self.pred_viewport,
                                                 video_rates=self.video_rates, tile_num_width=self.tile_num_width, tile_num_height=self.tile_num_height)
        self.tile_rates = list(self.tile_rates)
                
        selected_tile_sizes, selected_tile_quality, chunk_size, chunk_quality, download_time, \
        rebuffer_time, actual_viewport, over = self.simulator.simulate_download(self.tile_rates)
        qoe, qoe1, qoe2, qoe3, sample_weight, sample_weight_qoe = self.qoe_model.calculate_qoe_for_paas(qoe_weights_set=self.qoe_weights,
                                                                                                        actual_viewport=self.gt_viewport,
                                                                                                        tile_quality=selected_tile_quality,
                                                                                                        rebuffer_time=rebuffer_time,)
        sample_weight = np.array(sample_weight, dtype=np.float32)
        if self.mode == 'train':
            # reward = 0.5 * (qoe + sample_weight_qoe)
            reward = qoe
        else:
            reward = qoe
        self.log_qoe.append(float(qoe))
        self.log_qoe1.append(float(qoe1))
        self.log_qoe2.append(float(qoe2))
        self.log_qoe3.append(float(qoe3))

        self.past_throughputs = np.roll(self.past_throughputs, 1)
        self.past_throughputs[0, 0] = normalize_throughput(self.config, chunk_size / download_time)
        self.previous_pred_accuracy = np.roll(self.previous_pred_accuracy, 1)
        self.previous_pred_accuracy[0, 0] = self.last_chunk_accuracy
        self.buffer[0] = self.simulator.get_buffer_size()
        self.last_viewport_quality[0] = normalize_quality(self.config, sum(actual_viewport * selected_tile_quality) / sum(actual_viewport))
        
        if over:
            self.state.update({
                'throughput': self.past_throughputs,
                'buffer': self.buffer,
                'last_viewport_quality': self.last_viewport_quality,
                'chunk_bitrates': normalize_quality(self.config, self.next_chunk_quality),
                'pred_viewport': self.pred_viewport,
                'viewport_acc': self.previous_pred_accuracy,
                'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
                'sample_weight': normalize_qoe_weight(sample_weight),
                'sample_weight_qoe': sample_weight_qoe
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
                'buffer': self.buffer,
                'last_viewport_quality': self.last_viewport_quality,
                'chunk_bitrates': normalize_quality(self.config, self.next_chunk_quality),
                'pred_viewport': self.pred_viewport,
                'viewport_acc': self.previous_pred_accuracy,
                'qoe_weight': normalize_qoe_weight(self.current_qoe_weight),
                'sample_weight': normalize_qoe_weight(sample_weight),
                'sample_weight_qoe': sample_weight_qoe
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


def _test():
    import random
    from utils.common import get_config_from_yml

    log_dir = '/data/wuduo/2023_omnidirectional_vs/results/bitrate_selection/paas/Wu2017'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'testing.csv')
    config = get_config_from_yml()
    paas_env = PAASEnv(config, 'Wu2017', '4G', [(1, 1, 1)], log_path, config.startup_download)
    random.seed(0)
    tmp = paas_env.reset()
    paas_env.simulator.end_chunk = 11
    # paas_env.simulator.net_trace.trace_len = 2
    print(isinstance(paas_env, gym.Env))

    chunk, tile_id, over = 0, 0, False
    while chunk < 1000 and not over:
        action = random.randint(0, 4)
        state, reward, over, _ = paas_env.step(action)
        print('Chunk:', chunk)
        print('Reward:', reward)
        print('Throughputs:', state['throughput'])
        print('viewport_acc:', state['viewport_acc'])
        print('chunk_bitrates:', state['chunk_bitrates'])
        print('pred_viewport:', state['pred_viewport'])
        print('Buffer:', state['buffer'])
        print('last_viewport_quality:', state['last_viewport_quality'])
        chunk += 1
        if over:
            paas_env.reset()
            break


if __name__ == '__main__':
    _test()
