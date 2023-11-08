import torch
import torch.nn as nn


class FeatureNet(nn.Module):
    def __init__(self,pask_k, tile_total_num, num_rates, hidden_dim=128, device='cuda'):
        super(FeatureNet, self).__init__()
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
        self.conv1d6 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # rates inside
        self.conv1d7 = nn.Sequential(nn.Conv1d(1, hidden_dim, pask_k), nn.LeakyReLU(), nn.Flatten())  # rates outside
        self.fc1 = nn.Sequential(nn.Linear(1, hidden_dim), nn.LeakyReLU())  # buffer size
        self.fc2 = nn.Sequential(nn.Linear(3, hidden_dim), nn.LeakyReLU())  # qoe weight

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)


    def forward(self, observation):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        next_chunk_size = torch.from_numpy(observation['next_chunk_size']).to(self.device)
        next_chunk_quality = torch.from_numpy(observation['next_chunk_quality']).to(self.device)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)
        viewport_acc = torch.from_numpy(observation['viewport_acc']).to(self.device)
        rates_inside = torch.from_numpy(observation['rates_inside']).to(self.device)
        rates_outside = torch.from_numpy(observation['rates_outside']).to(self.device)
        buffer = torch.from_numpy(observation['buffer']).to(self.device)
        qoe_weight = torch.from_numpy(observation['qoe_weight']).to(self.device)

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(qoe_weight)

        features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
        features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
        features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
        features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
        features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
        features6, cell_states6 = self.lstm(self.conv1d6(rates_inside), cell_states5)
        features7, cell_states7 = self.lstm(self.conv1d7(rates_outside), cell_states6)
        features8, cell_states8 = self.lstm(self.fc1(buffer), cell_states7)
        features9, cell_states9 = self.lstm(self.fc2(qoe_weight), cell_states8)

        return features9


class FeatureNet2(nn.Module):
    def __init__(self,pask_k, tile_total_num, num_rates, hidden_dim=128, device='cuda', use_lstm=True):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
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

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)


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

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(qoe_weight)

        if self.use_lstm:
            features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
            features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
            features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
            features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
            features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
            features6, cell_states6 = self.lstm(self.conv1d6(past_viewport_qualities), cell_states5)
            features7, cell_states7 = self.lstm(self.conv1d7(past_quality_variances), cell_states6)
            features8, cell_states8 = self.lstm(self.conv1d8(past_rebuffering), cell_states7)
            features9, cell_states9 = self.lstm(self.fc1(buffer), cell_states8)
            features10, cell_states10 = self.lstm(self.fc2(qoe_weight), cell_states9)
        else:
            features10 = torch.cat([
                self.conv1d1(throuhgput),
                self.conv1d2(next_chunk_size),
                self.conv1d3(next_chunk_quality),
                self.conv1d4(pred_viewport),
                self.conv1d5(viewport_acc),
                self.conv1d6(past_viewport_qualities),
                self.conv1d7(past_quality_variances),
                self.conv1d8(past_rebuffering),
                self.fc1(buffer),
                self.fc2(qoe_weight),
            ], dim=-1)
        return features10


class Actor(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, action_space, device):
        super().__init__()
        self.feature_net = feature_net
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, action_space).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features = self.feature_net(batch).to(self.device)
        # logits = torch.softmax(self.out(features), dim=1)
        logits = self.out(self.fc(features))  # residual
        return logits, state


class Critic(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, 1).to(device)
        self.device = device

    def forward(self, batch, state=None, info={}):
        features = self.feature_net(batch).to(self.device)
        logits = self.out(self.fc(features))  # residual
        return logits


class FeatureNet3(nn.Module):
    def __init__(self,pask_k, tile_total_num, num_rates, hidden_dim=128, device='cuda', use_lstm=True):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
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

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)


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

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(qoe_weight)

        qoe_weight_features = self.fc2(qoe_weight)
        if self.use_lstm:
            features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
            features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
            features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
            features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
            features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
            features6, cell_states6 = self.lstm(self.conv1d6(past_viewport_qualities), cell_states5)
            features7, cell_states7 = self.lstm(self.conv1d7(past_quality_variances), cell_states6)
            features8, cell_states8 = self.lstm(self.conv1d8(past_rebuffering), cell_states7)
            features9, cell_states9 = self.lstm(self.fc1(buffer), cell_states8)
            features10, cell_states10 = self.lstm(qoe_weight_features, cell_states9)
        else:
            features10 = torch.cat([
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
        return features10, qoe_weight_features


class Actor3(nn.Module):
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


class Critic3(nn.Module):
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
    def __init__(self, pask_k, tile_total_num, num_rates, action_space, hidden_dim=128, device='cuda', use_lstm=True):
        super(QoEIdentifierFeatureNet, self).__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
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

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)
    
    def forward(self, observation, action_one_hot):
        throuhgput = torch.from_numpy(observation['throughput']).to(self.device)
        next_chunk_size = torch.from_numpy(observation['next_chunk_size']).to(self.device)
        next_chunk_quality = torch.from_numpy(observation['next_chunk_quality']).to(self.device)
        pred_viewport = torch.from_numpy(observation['pred_viewport']).to(self.device)
        viewport_acc = torch.from_numpy(observation['viewport_acc']).to(self.device)
        past_viewport_qualities = torch.from_numpy(observation['past_viewport_qualities']).to(self.device)
        past_quality_variances = torch.from_numpy(observation['past_quality_variances']).to(self.device)
        past_rebuffering = torch.from_numpy(observation['past_rebuffering']).to(self.device)
        # rates_inside = torch.from_numpy(observation['rates_inside']).to(self.device)
        # rates_outside = torch.from_numpy(observation['rates_outside']).to(self.device)
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
            # rates_inside = rates_inside.unsqueeze(0)
            # rates_outside = rates_outside.unsqueeze(0)
            buffer = buffer.unsqueeze(0)
            action_one_hot = action_one_hot.unsqueeze(0)

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(action_one_hot)

        if self.use_lstm:
            features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
            features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
            features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
            features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
            features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
            features6, cell_states6 = self.lstm(self.conv1d6(past_viewport_qualities), cell_states5)
            features7, cell_states7 = self.lstm(self.conv1d7(past_quality_variances), cell_states6)
            features8, cell_states8 = self.lstm(self.conv1d8(past_rebuffering), cell_states7)
            # features6, cell_states6 = self.lstm(self.conv1d6(rates_inside), cell_states5)
            # features7, cell_states7 = self.lstm(self.conv1d7(rates_outside), cell_states6)
            features9, cell_states9 = self.lstm(self.fc1(buffer), cell_states8)
            features10, cell_states10 = self.lstm(self.fc2(action_one_hot), cell_states9)
        else:
            features10 = torch.cat([
                self.conv1d1(throuhgput),
                self.conv1d2(next_chunk_size),
                self.conv1d3(next_chunk_quality),
                self.conv1d4(pred_viewport),
                self.conv1d5(viewport_acc),
                self.conv1d6(past_viewport_qualities),
                self.conv1d7(past_quality_variances),
                self.conv1d8(past_rebuffering),
                self.fc1(buffer),
                self.fc2(action_one_hot),
            ])
        return features10


class QoEIdentifierFeatureNet2(nn.Module):
    def __init__(self, pask_k, tile_total_num, num_rates, action_space, hidden_dim=128, device='cuda', use_lstm=True):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
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

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)
    
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

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(action_one_hot)
        if self.use_lstm:
            features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
            features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
            features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
            features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
            features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
            features6, cell_states6 = self.lstm(self.conv1d6(past_viewport_qualities), cell_states5)
            features7, cell_states7 = self.lstm(self.conv1d7(past_quality_variances), cell_states6)
            features8, cell_states8 = self.lstm(self.conv1d8(past_rebuffering), cell_states7)
            features9, cell_states9 = self.lstm(self.fc1(buffer), cell_states8)
            features10, cell_states10 = self.lstm(self.fc2(action_one_hot), cell_states9)
        else:
            features10 = torch.cat([
                self.conv1d1(throuhgput),
                self.conv1d2(next_chunk_size),
                self.conv1d3(next_chunk_quality),
                self.conv1d4(pred_viewport),
                self.conv1d5(viewport_acc),
                self.conv1d6(past_viewport_qualities),
                self.conv1d7(past_quality_variances),
                self.conv1d8(past_rebuffering),
                self.fc1(buffer),
                self.fc2(action_one_hot),
            ], dim=-1)
        return features10


class QoEIdentifier(nn.Module):
    def __init__(self, feature_net, feature_dim, hidden_dim, device):
        super().__init__()
        self.feature_net = feature_net
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.out = nn.Linear(hidden_dim, 3).to(device)
        self.device = device

    def forward(self, observation, action_one_host):
        features = self.feature_net(observation, action_one_host).to(self.device)
        logits = self.out(self.fc(features))  # residual
        logits = torch.sigmoid(logits)
        return logits


class QoEIdentifierFeatureNet3(nn.Module):
    def __init__(self, pask_k, tile_total_num, num_rates, action_space, hidden_dim=128, device='cuda', use_lstm=True):
        super().__init__()
        self.past_k = pask_k
        self.tile_total_num = tile_total_num
        self.num_rates = num_rates
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
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

        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, batch_first=True)
    
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

        # t1 = self.conv1d1(throuhgput)
        # t2 = self.conv1d2(next_chunk_size)
        # t3 = self.conv1d3(next_chunk_quality)
        # t4 = self.conv1d4(pred_viewport)
        # t5 = self.conv1d5(viewport_acc)
        # t6 = self.conv1d6(rates_inside)
        # t7 = self.conv1d7(rates_outside)
        # t8 = self.fc1(buffer)
        # t9 = self.fc2(action_one_hot)
        action_one_hot_features = self.fc2(action_one_hot)
        if self.use_lstm:
            features1, cell_states1 = self.lstm(self.conv1d1(throuhgput))
            features2, cell_states2 = self.lstm(self.conv1d2(next_chunk_size), cell_states1)
            features3, cell_states3 = self.lstm(self.conv1d3(next_chunk_quality), cell_states2)
            features4, cell_states4 = self.lstm(self.conv1d4(pred_viewport), cell_states3)
            features5, cell_states5 = self.lstm(self.conv1d5(viewport_acc), cell_states4)
            features6, cell_states6 = self.lstm(self.conv1d6(past_viewport_qualities), cell_states5)
            features7, cell_states7 = self.lstm(self.conv1d7(past_quality_variances), cell_states6)
            features8, cell_states8 = self.lstm(self.conv1d8(past_rebuffering), cell_states7)
            features9, cell_states9 = self.lstm(self.fc1(buffer), cell_states8)
            features10, cell_states10 = self.lstm(action_one_hot_features, cell_states9)
        else:
            features10 = torch.cat([
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
        return features10, action_one_hot_features


class QoEIdentifier3(nn.Module):
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
