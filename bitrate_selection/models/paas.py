import torch
import torch.nn as nn
import numpy as np


class FeatureNet(nn.Module):
    def __init__(self, pask_k, tile_total_num, num_rates, device='cuda'):
        super(FeatureNet, self).__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.device = device

        self.conv1d_1 = nn.Sequential(nn.Conv1d(1, 128, pask_k), nn.LeakyReLU(), nn.Flatten())  # throughput
        self.conv1d_2 = nn.Sequential(nn.Conv1d(1, 128, tile_total_num * num_rates), nn.LeakyReLU(), nn.Flatten())  # next chunk size
        self.fc1 = nn.Sequential(nn.Linear(1, 128), nn.LeakyReLU())  # buffer size
        self.fc2 = nn.Sequential(nn.Linear(1, 128), nn.LeakyReLU())  # last_viewport_quality
        self.conv1d_2 = nn.Sequential(nn.Conv1d(1, 128, tile_total_num * num_rates), nn.LeakyReLU(), nn.Flatten())  # chunk_bitrate
        self.fc3 = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU())  # pred_viewport
        self.conv1d_5 = nn.Sequential(nn.Conv1d(1, 128, 8), nn.LeakyReLU(), nn.Flatten())  # viewport_acc
        self.conv1d_6 = nn.Sequential(nn.Conv1d(1, 128, 3), nn.LeakyReLU(), nn.Flatten())  # qoe_weight

        
    def forward(self, observation):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        buffer = torch.from_numpy(observation['buffer']).to(self.device)
        last_viewport_quality = torch.from_numpy(observation['last_viewport_quality']).to(self.device)
        chunk_bitrates = torch.from_numpy(observation['chunk_bitrates']).to(self.device).reshape(-1, 1, 5*64)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)
        viewport_acc = torch.from_numpy(observation['viewport_acc']).to(self.device)
        qoe_weight = torch.from_numpy(observation['qoe_weight']).to(self.device).unsqueeze(1)

        # t1 = self.conv1d_1(throuhgput)
        # t2 = self.fc1(buffer)
        # t3 = self.fc2(last_viewport_quality)
        # t4 = self.conv1d_2(chunk_bitrates)
        # t5 = self.fc3(pred_viewport)
        # t8 = self.conv1d_5(viewport_acc)
        # t9 = self.conv1d_6(qoe_weight)

        logits = torch.cat([
            self.conv1d_1(throuhgput),
            self.fc1(buffer),
            self.fc2(last_viewport_quality),
            self.conv1d_2(chunk_bitrates),
            self.fc3(pred_viewport),
            self.conv1d_5(viewport_acc),
            self.conv1d_6(qoe_weight)
        ], dim=-1)
        return logits


class ForwardNet(nn.Module):
    def __init__(self, action_space, device='cuda'):
        super(ForwardNet, self).__init__()
        self.device = device
        self.fc_large = nn.Sequential(nn.Linear(128 * 7, 1024), nn.LeakyReLU(), nn.LayerNorm(1024))
        self.fc_small = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU(), nn.LayerNorm(256))
        self.fc_out = nn.Linear(256, action_space)

    def forward(self, features):
        features1 = self.fc_large(features)
        features2 = self.fc_small(features1)
        logits = self.fc_out(features2)
        return logits


class PAASNet(nn.Module):
    def __init__(self, config, device='cuda'):
        super(PAASNet, self).__init__()
        self.device = device
        self.config = config
        self.feature_net = FeatureNet(config.past_k, config.tile_total_num, len(config.video_rates), device).to(device)
        self.feed_forward = ForwardNet(config.action_space, device).to(device)
        self._init_params()

    def forward(self, batch, state=None, info={}):
        logits = self.feed_forward(self.feature_net(batch))
        return logits, state

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            q, _ = self.forward(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.config.action_space - 1)
            return action

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=1, std=10)
                nn.init.normal_(m.bias, 1)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=1, std=10)
                nn.init.normal_(m.bias, 1)

    def learn(self, replay_buffer, total_steps):
        # use dynamic preference training
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # Use target_net to estimate the q_target
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            else:
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).max(dim=-1)[0]  # shape：(batch_size,)

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)
