import torch
import torch.nn as nn
import numpy as np


class FeatureNet(nn.Module):
    def __init__(self, pask_k, tile_total_num, num_rates, device='cuda'):
        super(FeatureNet, self).__init__()
        self.device = device

        self.conv1d_1 = nn.Sequential(nn.Conv1d(1, 128, pask_k), nn.LeakyReLU(), nn.Flatten())  # throughput
        self.conv1d_2 = nn.Sequential(nn.Conv1d(1, 128, tile_total_num * num_rates), nn.LeakyReLU(), nn.Flatten())  # chunk sizes
        self.fc1 = nn.Sequential(nn.Linear(1, 128), nn.LeakyReLU())  # rebuffering time
        self.fc2 = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU())  # last bitrates
        self.fc3 = nn.Sequential(nn.Linear(1, 128), nn.LeakyReLU())  # chunk remain
        self.fc4 = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU())  # predicted viewport

        
    def forward(self, observation):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        next_chunk_size = torch.from_numpy(observation['chunk_sizes']).to(self.device).reshape(-1, 1, 5 * 64)
        rebuffer = torch.from_numpy(observation['rebuffer']).to(self.device)
        last_bitrates = torch.from_numpy(observation['last_bitrates']).to(self.device)
        chunk_remain = torch.from_numpy(observation['chunk_remain']).to(self.device)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)

        # t1 = self.conv1d_1(throuhgput)
        # t2 = self.conv1d_2(next_chunk_size)
        # t3 = self.fc1(rebuffer)
        # t4 = self.fc2(last_bitrates)
        # t5 = self.fc3(chunk_remain)
        # t6 = self.fc4(pred_viewport)

        logits = torch.cat([
            self.conv1d_1(throuhgput),
            self.conv1d_2(next_chunk_size),
            self.fc1(rebuffer),
            self.fc2(last_bitrates),
            self.fc3(chunk_remain),
            self.fc4(pred_viewport),
        ], dim=-1)
        return logits


class Actor(nn.Module):
    def __init__(self, feature_net, feature_dim, action_space, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(*[nn.Linear(feature_dim, 128), nn.LeakyReLU()]).to(device)
        self.out = nn.Linear(128, action_space).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features = self.feature_net(batch).to(self.device)
        logits = torch.softmax(self.out(self.fc(features)), dim=1)
        return logits, state


class Critic(nn.Module):
    def __init__(self, feature_net, feature_dim, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(*[nn.Linear(feature_dim, 128), nn.LeakyReLU()]).to(device)
        self.out = nn.Linear(128, 1).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features = self.feature_net(batch).to(self.device)
        logits = self.out(self.fc(features))
        return logits
