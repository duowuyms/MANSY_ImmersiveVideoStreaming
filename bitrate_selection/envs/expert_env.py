import os
import gym
import pickle
import numpy as np
from tqdm import trange
from simulators.simulator import ExpertSimulator
from utils.common import (normalize_quality, normalize_size, normalize_throughput, normalize_qoe_weight, 
                          action2rates, rates2action, allocate_tile_rates)
from utils.qoe import QoEModelExpert


class ExpertEnv(gym.Env):
    """
    A environment class for an MPC-based expert.
    You may use it to develop your own expert for imitation learning.
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    init = False
    all_video_rates = []
    all_videos = []
    all_users = []
    chunk_gt_viewport_qualities = {}
    chunk_pred_viewport_qualities = {}
    chunk_gt_intra_quality_variance = {}
    chunk_pred_intra_quality_variance = {}
    chunk_gt_sizes = {}
    chunk_pred_sizes = {}

    def __init__(self, config, dataset, network_dataset, qoe_weights, samples, demos_dir, cache_path, log_path, startup_download, horizon,
                 refresh_cache=True, mode='train', seed=0, device='cpu'):
        self.config = config
        self.dataset = dataset
        self.network_dataset = network_dataset
        self.qoe_weights = qoe_weights
        self.samples = samples
        self.demos_dir = demos_dir
        self.log_path = log_path
        self.startup_download = startup_download
        self.horizon = horizon
        self.video_rates = config.video_rates
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
        self.last_chunk_accuracy = 0

        self.next_chunk_size = None
        self.next_chunk_quality = None

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

        if not self.init:
            self.all_bitrates = self._proflie_all_possible_bitrates()
            for split in ['train', 'valid', 'test']:
                self.all_videos += config.video_split[dataset][split]
                self.all_users += config.user_split[dataset][split]
            self.all_videos = list(set(self.all_videos))
            self.all_users = list(set(self.all_users))
            
            if not os.path.exists(cache_path) or refresh_cache:
                self._profile_viewport_qualities_sizes()
                pickle.dump([self.chunk_gt_viewport_qualities, self.chunk_pred_viewport_qualities,
                             self.chunk_gt_intra_quality_variance, self.chunk_pred_intra_quality_variance,
                             self.chunk_gt_sizes, self.chunk_pred_sizes], open(cache_path, 'wb'))
                print('Save expert cache at', cache_path)
            else:
                self.chunk_gt_viewport_qualities, self.chunk_pred_viewport_qualities, \
                    self.chunk_gt_intra_quality_variance, self.chunk_pred_intra_quality_variance, \
                    self.chunk_gt_sizes, self.chunk_pred_sizes = pickle.load(open(cache_path, 'rb'))
                print('Load expert cache from', cache_path)
            self.init = True

    def _proflie_all_possible_bitrates(self):
        all_bitrates = []
        action_space = self.config.action_space
        for i in range(action_space ** self.horizon):
            bitrates = []
            tmp = i
            for j in range(self.horizon):
                action = tmp % action_space
                rate_in, rate_out = action2rates(action)
                bitrates.append((rate_in, rate_out))
                tmp = tmp // action_space
            all_bitrates.append(bitrates)
        return all_bitrates

    def _profile_viewport_qualities_sizes(self):
        video_user_pairs = []
        for video in self.all_videos:
            for user in self.all_users:
                video_user_pairs.append((video, user))
                self.chunk_gt_viewport_qualities[video, user] = {}
                self.chunk_pred_viewport_qualities[video, user] = {}
                self.chunk_gt_intra_quality_variance[video, user] = {}
                self.chunk_pred_intra_quality_variance[video, user] = {}
                self.chunk_gt_sizes[video, user] = {}
                self.chunk_pred_sizes[video, user] = {}                

        for i, (video, user) in enumerate(video_user_pairs):
            simulator = ExpertSimulator(config=self.config, dataset=self.dataset, video=video,
                                        network_dataset=self.network_dataset, user=user, trace=0,  # a random trace is just fine
                                        startup_download=self.startup_download)
            chunk = simulator.next_chunk
            while chunk <= simulator.end_chunk:
                tmp_gt_viewport_qualities = {}
                tmp_pred_viewport_qualities = {}
                tmp_gt_intra_quality_variance = {}
                tmp_pred_intra_quality_variance = {}
                tmp_gt_chunk_sizes = {}
                tmp_pred_chunk_sizes = {}
                gt_viewport, pred_viewport, _ = simulator.get_viewport(chunk)
                for action in range(self.config.action_space):
                    rate_in, rate_out = action2rates(action)

                    # calculate chunk size and viewport quality of gt viewport
                    tile_rates, _ = allocate_tile_rates(rate_version_in=rate_in, rate_version_out=rate_out,
                                                        pred_viewport=gt_viewport, video_rates=self.video_rates,
                                                        tile_num_width=self.tile_num_width, tile_num_height=self.tile_num_height)
                    tile_rates = list(tile_rates)
                    gt_chunk_size, gt_viewport_quality, gt_tile_quality = simulator.calculate_chunk_size_and_quality(chunk, tile_rates, gt_viewport)
                    tmp_gt_viewport_qualities[(rate_in, rate_out)] = gt_viewport_quality
                    tmp_gt_intra_quality_variance[(rate_in, rate_out)] = sum(gt_viewport * abs(gt_tile_quality - gt_viewport_quality)) / sum(gt_viewport)
                    tmp_gt_chunk_sizes[(rate_in, rate_out)] = gt_chunk_size

                    # calculate chunk size and viewport quality of predicted viewport
                    tile_rates, _ = allocate_tile_rates(rate_version_in=rate_in, rate_version_out=rate_out,
                                                        pred_viewport=pred_viewport, video_rates=self.video_rates,
                                                        tile_num_width=self.tile_num_width, tile_num_height=self.tile_num_height)
                    tile_rates = list(tile_rates)
                    pred_chunk_size, pred_viewport_quality, pred_tile_quality = simulator.calculate_chunk_size_and_quality(chunk, tile_rates, gt_viewport)
                    tmp_pred_viewport_qualities[(rate_in, rate_out)] = pred_viewport_quality
                    tmp_pred_intra_quality_variance[(rate_in, rate_out)] = sum(gt_viewport * abs(pred_tile_quality - pred_viewport_quality)) / sum(gt_viewport)
                    tmp_pred_chunk_sizes[(rate_in, rate_out)] = pred_chunk_size

                self.chunk_gt_viewport_qualities[video, user][chunk] = tmp_gt_viewport_qualities
                self.chunk_pred_viewport_qualities[video, user][chunk] = tmp_pred_viewport_qualities
                self.chunk_gt_intra_quality_variance[video, user][chunk] = tmp_gt_intra_quality_variance
                self.chunk_pred_intra_quality_variance[video, user][chunk] = tmp_pred_intra_quality_variance
                self.chunk_gt_sizes[video, user][chunk] = tmp_gt_chunk_sizes
                self.chunk_pred_sizes[video, user][chunk] = tmp_pred_chunk_sizes
                chunk += 1
            print(f'Cache of video-{video}, user-{user} done! ({i + 1}/{len(video_user_pairs)})')
    
    def reset(self):
        self.sample_id = (self.sample_id + 1) % self.sample_len
        self.current_video = self.videos[self.samples[self.sample_id][0]]
        self.current_user = self.users[self.samples[self.sample_id][1]]
        self.current_trace = self.traces[self.samples[self.sample_id][2]]
        self.current_qoe_weight = np.array(self.qoe_weights[self.samples[self.sample_id][3]], dtype=np.float32)

        self.simulator = ExpertSimulator(config=self.config,
                                         dataset=self.dataset,
                                         network_dataset=self.network_dataset,
                                         video=self.current_video,
                                         user=self.current_user,
                                         trace=self.current_trace,
                                         startup_download=self.startup_download)
        self.qoe_model = QoEModelExpert(self.config, *self.current_qoe_weight)

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
        
        reward = qoe
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

        action_one_hot = np.zeros(self.config.action_space, dtype=np.float32)
        action_one_hot[action] = 1.

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
    
    def choose_action(self):
        best_action_reward = float('-inf')
        best_index = 0

        horizon = min(self.horizon, self.simulator.end_chunk - self.simulator.next_chunk + 1)

        chunk_gt_viewport_qualities = []
        chunk_pred_viewport_qualities = []
        chunk_gt_intra_quality_variance = []
        chunk_pred_intra_quality_variance = []
        chunk_gt_sizes = []
        chunk_pred_sizes = []
        for t in range(horizon):
            current_chunk = self.simulator.next_chunk + t
            chunk_gt_viewport_qualities.append(self.chunk_gt_viewport_qualities[self.current_video, self.current_user][current_chunk])
            chunk_pred_viewport_qualities.append(self.chunk_pred_viewport_qualities[self.current_video, self.current_user][current_chunk])
            chunk_gt_intra_quality_variance.append(self.chunk_gt_intra_quality_variance[self.current_video, self.current_user][current_chunk])
            chunk_pred_intra_quality_variance.append(self.chunk_pred_intra_quality_variance[self.current_video, self.current_user][current_chunk])
            chunk_gt_sizes.append(self.chunk_gt_sizes[self.current_video, self.current_user][current_chunk])
            chunk_pred_sizes.append(self.chunk_pred_sizes[self.current_video, self.current_user][current_chunk])
        
        for i in range(len(self.all_bitrates)):
            # for debug (start)
            record_chunk = self.simulator.next_chunk
            record_buf_size = self.simulator.buffer.buf_size
            record_cur_idx = self.simulator.net_trace.cur_idx
            record_cur_time = self.simulator.net_trace.cur_time
            record_prev_viewport_quality = self.qoe_model.prev_viewport_quality
            # for debug (end)
            qoe_sum = 0
            prev_viewport_quality = self.qoe_model.prev_viewport_quality
            for t in range(horizon):
                start = False
                end = False
                if t == 0:
                    start = True
                if t == horizon - 1:
                    end = True
                rate_in, rate_out = self.all_bitrates[i][t]

                viewport_quality = chunk_pred_viewport_qualities[t][(rate_in, rate_out)]
                intra_viewport_variance = chunk_pred_intra_quality_variance[t][(rate_in, rate_out)]
                chunk_size = chunk_pred_sizes[t][(rate_in, rate_out)]
                
                download_time, rebuffer_time, over = \
                    self.simulator.virtual_simulate_download_with_chunk_size(chunk_size, start, end)
                qoe, qoe1, qoe2, qoe3, prev_viewport_quality = self.qoe_model.calculate_qoe_with_given_quality(
                    viewport_quality=viewport_quality,
                    prev_viewport_quality=prev_viewport_quality,
                    intra_viewport_quality_variance=intra_viewport_variance,
                    rebuffer_time=rebuffer_time,
                )
                qoe_sum += qoe 
            if best_action_reward < qoe_sum:
                best_action_reward = qoe_sum
                best_index = i
            self.simulator.next_chunk -= horizon
            assert record_chunk == self.simulator.next_chunk
            assert record_prev_viewport_quality == self.qoe_model.prev_viewport_quality
            assert record_cur_idx == self.simulator.net_trace.cur_idx
            assert record_cur_time == self.simulator.net_trace.cur_time
            assert record_buf_size == self.simulator.buffer.buf_size
        best_action = rates2action(rate_in=self.all_bitrates[best_index][0][0], 
                                   rate_out=self.all_bitrates[best_index][0][1])
        return best_action

