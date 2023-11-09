import math
import random
import numpy as np
import torch
import torch.nn as nn
from models.customized_transformer import Transformer
from utils.common import mean_square_error, to_position_normalized_cartesian


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ViewportEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(self.in_channels, embedding_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.linear(x[:, i, :]).unsqueeze(1))
        return torch.cat(batch_embeddings, dim=1)


class ViewportTransformerMTIO(nn.Module):
    def __init__(self, in_channel, fut_window, d_model, dim_feedforward, num_head=3, num_encoder_layers=2, 
                 num_decoder_layers=2, batch_first=True, dropout=0.2, device='cuda', repeat_prob=0.5, seed=1):
        super().__init__()

        self.in_channel = in_channel
        self.num_head = num_head
        self.fut_window = fut_window
        self.embedding = ViewportEmbedding(in_channels=in_channel * num_head, embedding_dim=d_model).to(device)
        self.transformer = Transformer(d_model=d_model, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                       batch_first=batch_first)
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=dropout).to(device)
        self.predictor = nn.Sequential(nn.Linear(d_model, in_channel * num_head), nn.Sigmoid())
        self.device = device
        self.repeat_prob = repeat_prob
        self.seed = seed

    def forward(self, history, current, future):
        """
        Forward pass through the network. Encode the past trajectory and decode the future trajectory.
        :param history: (Tensor) Historical trajectories [batch_size x M x in_channels]
        :param current: (Tensor) Current coordinates [batch_size x 1 x in_channels]
        :param future: (Tensor) Ground truth future trajectories [batch_size * H x in_channels]
        """        
        multi_history = [history]
        multi_current = [current]
        multi_future = [future]
        # repeat_prob is a trick borrowed from the following paper:
        # https://arxiv.org/abs/2010.06610
        if random.random() < self.repeat_prob:
            multi_history = [history for _ in range(self.num_head)]
            multi_current = [current for _ in range(self.num_head)]
            multi_future = [future for _ in range(self.num_head)]
        else:
            for _ in range(self.num_head - 1):
                indices = np.arange(len(history))
                np.random.shuffle(indices)
                multi_history.append(history[indices])
                multi_current.append(current[indices])
                multi_future.append(future[indices])
        multi_history = torch.cat(multi_history, dim=-1)
        multi_current = torch.cat(multi_current, dim=-1)
        multi_future = torch.cat(multi_future, dim=-1)
        pred = self._process_src_current(src=multi_history, current=multi_current)
        return pred, multi_future
        
    def loss_function(self, pred, gt):
        """
        Compute the loss function.
        :param pred: predicted viewports
        :param gt: ground-truth viewports
        """
        loss = 0.
        for i in range(self.num_head):
            loss += torch.mean(mean_square_error(pred[:, :, i * self.in_channel:(i + 1) * self.in_channel],
                                                 gt[:, :, i * self.in_channel:(i + 1) * self.in_channel]))
        return loss

    def sample(self, history, current):
        """
        Sample predicted viewports.
        :param history: (Tensor) Historical trajectory to initialize decoder state [batch_size x M x in_channels]
        :param current: (Tensor) Current coordinates to initialize decoder input [batch_size x 1 x in_channels]
        """
        multi_history = [history for _ in range(self.num_head)]
        multi_current = [current for _ in range(self.num_head)]
        multi_history = torch.cat(multi_history, dim=-1)
        multi_current = torch.cat(multi_current, dim=-1)

        src = multi_history
        tgt = multi_current
        ensemble_viewports = []
        for i in range(self.fut_window):
            out = self._process_src_tgt(src, tgt)
            pred = self.predictor(out[:, -1]).unsqueeze(1)
            tgt = torch.cat([tgt, pred], dim=1)

            ensembled_pred = []
            for i in range(self.in_channel):
                indices = [i + j * self.in_channel for j in range(self.num_head)]
                ensembled_pred.append(torch.sum(pred[:, :, indices], dim=-1, keepdim=True) / self.num_head)
            ensembled_pred = torch.cat(ensembled_pred, dim=-1)
            ensemble_viewports.append(ensembled_pred)
        ensemble_viewports = torch.cat(ensemble_viewports, dim=1)
        ensemble_viewports = to_position_normalized_cartesian(ensemble_viewports)
        return ensemble_viewports

    def _process_src_tgt(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)

        # encode src and tgt
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # add positional information to src and tgt
        src = self.positional_embedding(src)
        tgt = self.positional_embedding(tgt)

        # feed to transformer
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return out

    def _process_src_current(self, src, current):
        # encode src
        src = self.embedding(src)
        src = self.positional_embedding(src)
        # create memory
        memory = self.transformer.encode(src)

        tgt = current
        for i in range(self.fut_window):
            tgt_embed = self.embedding(tgt)
            tgt_embed = self.positional_embedding(tgt_embed)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.shape[1]).to(self.device)
            out = self.transformer.decode(tgt_embed, memory, tgt_mask=tgt_mask)
            pred = self.predictor(out[:, -1])
            tgt = torch.cat([tgt, pred.unsqueeze(1)], dim=1)
        pred = tgt[:, 1:]
        return pred
