import torch
import torch.nn as nn


class FeatureNet(nn.Module):
    def __init__(self,pask_k, tile_total_num, num_rates, hidden_dim=128, device='cuda'):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.device = device

        self.conv1d1 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # throughput
        self.conv1d2 = nn.Sequential(nn.Conv1d(num_rates, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # next chunk size
        self.conv1d3 = nn.Sequential(nn.Conv1d(num_rates, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # next chunk quality
        self.conv1d4 = nn.Sequential(nn.Conv1d(1, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # pred viewport
        self.conv1d5 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past viewport accuracy
        self.conv1d6 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past viewport quality
        self.conv1d7 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past quality variances
        self.conv1d8 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past rebuffering
        self.fc1 = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU())  # buffer size
        self.fc2 = nn.Sequential(nn.Linear(3, hidden_dim), nn.LeakyReLU())  # qoe weight


    def forward(self, observation):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        next_chunk_size = torch.from_numpy(observation['next_chunk_size']).to(self.device)
        next_chunk_quality = torch.from_numpy(observation['next_chunk_quality']).to(self.device)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)
        viewport_acc = torch.from_numpy(observation['viewport_acc']).to(self.device)
        past_viewport_qualities = torch.from_numpy(observation['past_viewport_qualities']).to(self.device)
        past_quality_variances = torch.from_numpy(observation['past_quality_variances']).to(self.device)
        past_rebuffering = torch.from_numpy(observation['past_rebuffering']).to(self.device)
        buffer = torch.from_numpy(observation['buffer']).to(self.device)
        qoe_weight = torch.from_numpy(observation['qoe_weight']).to(self.device)

        qoe_weight_features = self.fc2(qoe_weight)
        features = torch.cat([
            self.conv1d1(throuhgput),
            self.conv1d2(next_chunk_size),
            self.conv1d3(next_chunk_quality),
            self.conv1d4(pred_viewport),
            self.conv1d5(viewport_acc),
            self.conv1d6(past_viewport_qualities),
            self.conv1d7(past_quality_variances),
            self.conv1d8(past_rebuffering),
            self.fc1(buffer),
            qoe_weight_features,
        ], dim=-1)
        return features, qoe_weight_features


class Actor(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, action_space, device):
        super().__init__()
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, action_space).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features, qoe_features = self.feature_net(batch)
        logits = self.out(self.fc(features) + qoe_features)  # residual
        return logits, state


class Critic(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, 1).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features, qoe_features = self.feature_net(batch)
        logits = self.out(self.fc(features) + qoe_features)  # residual
        return logits


class QoEIdentifierFeatureNet(nn.Module):
    def __init__(self, pask_k, tile_total_num, num_rates, action_space, hidden_dim=128, device='cuda'):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.device = device

        self.conv1d1 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # throughput
        self.conv1d2 = nn.Sequential(nn.Conv1d(num_rates, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # next chunk size
        self.conv1d3 = nn.Sequential(nn.Conv1d(num_rates, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # next chunk quality
        self.conv1d4 = nn.Sequential(nn.Conv1d(1, hidden_dim, tile_total_num), nn.LeakyReLU(), nn.Flatten())  # pred viewport
        self.conv1d5 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past viewport accuracy
        self.conv1d6 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past viewport quality
        self.conv1d7 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past quality variances
        self.conv1d8 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # past rebuffering
        self.fc1 = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU())  # buffer size
        self.fc2 = nn.Sequential(nn.Linear(action_space, hidden_dim), nn.LeakyReLU())  # action one-hot vector
    
    def forward(self, observation, action_one_hot):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        next_chunk_size = torch.from_numpy(observation['next_chunk_size']).to(self.device)
        next_chunk_quality = torch.from_numpy(observation['next_chunk_quality']).to(self.device)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)
        viewport_acc = torch.from_numpy(observation['viewport_acc']).to(self.device)
        past_viewport_qualities = torch.from_numpy(observation['past_viewport_qualities']).to(self.device)
        past_quality_variances = torch.from_numpy(observation['past_quality_variances']).to(self.device)
        past_rebuffering = torch.from_numpy(observation['past_rebuffering']).to(self.device)
        buffer = torch.from_numpy(observation['buffer']).to(self.device)
        action_one_hot = torch.from_numpy(action_one_hot).to(self.device)

        if len(throuhgput.shape) == 2:
            throuhgput = throuhgput.unsqueeze(0)
            next_chunk_size = next_chunk_size.unsqueeze(0)
            next_chunk_quality = next_chunk_quality.unsqueeze(0)
            pred_viewport = pred_viewport.unsqueeze(0)
            viewport_acc = viewport_acc.unsqueeze(0)
            past_viewport_qualities = past_viewport_qualities.unsqueeze(0)
            past_quality_variances = past_quality_variances.unsqueeze(0)
            past_rebuffering = past_rebuffering.unsqueeze(0)
            buffer = buffer.unsqueeze(0)
            action_one_hot = action_one_hot.unsqueeze(0)

        action_one_hot_features = self.fc2(action_one_hot)
        features = torch.cat([
            self.conv1d1(throuhgput),
            self.conv1d2(next_chunk_size),
            self.conv1d3(next_chunk_quality),
            self.conv1d4(pred_viewport),
            self.conv1d5(viewport_acc),
            self.conv1d6(past_viewport_qualities),
            self.conv1d7(past_quality_variances),
            self.conv1d8(past_rebuffering),
            self.fc1(buffer),
            action_one_hot_features,
        ], dim=-1)
        return features, action_one_hot_features


class QoEIdentifier(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, 3).to(device)
        self.device = device

    def forward(self, observation, action_one_host):
        features, action_one_hot_features = self.feature_net(observation, action_one_host)
        logits = self.out(self.fc(features) + action_one_hot_features)  # residual
        logits = torch.sigmoid(logits)
        return logits
