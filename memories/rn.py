import torch
import torch.nn as nn


class RN(nn.Module):

    def __init__(self, input_size, g_size, f_size, out_size):
        super(RN, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(input_size, g_size),
            nn.Linear(g_size, g_size),
            nn.ReLU())

        self.f = nn.Sequential(
            nn.Linear(g_size, f_size),
            nn.Linear(f_size, f_size),
            nn.ReLU())

    def forward(self, input):
        g_out = torch.sum(self.g(input), 0)
        return self.f(g_out)
