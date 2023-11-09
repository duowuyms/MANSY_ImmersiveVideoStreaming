import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LinearRegression as LR


class LinearRegression(nn.Module):
    def __init__(self, fut_window, device='cpu'):
        super().__init__()
        self.fut_window = fut_window
        self.device = device

    def forward(self):
        pass

    def sample(self, history, current):
        merge = torch.cat([history, current], dim=1)
        batch_size = merge.shape[0]
        past_sample_length = merge.shape[1]
        past_window = np.arange(past_sample_length).reshape(-1, 1)
        future_window = np.arange(past_sample_length, past_sample_length + self.fut_window).reshape(-1, 1)
        samples = torch.zeros(batch_size, self.fut_window, 2)
        for i in range(batch_size):
            x_data = merge[i, :, 0].numpy()
            y_data = merge[i, :, 1].numpy()
            predictor_x = LR(fit_intercept=True).fit(past_window, x_data.reshape(-1, 1))
            predictor_y = LR(fit_intercept=True).fit(past_window, y_data.reshape(-1, 1))
            x_pred = predictor_x.predict(future_window)
            y_pred = predictor_y.predict(future_window)
            x_pred = torch.from_numpy(x_pred).reshape(-1, 1)
            y_pred = torch.from_numpy(y_pred).reshape(-1, 1)
            samples[i] = torch.cat([x_pred, y_pred], dim=-1)
        return samples

