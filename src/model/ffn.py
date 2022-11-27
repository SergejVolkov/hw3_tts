import torch.nn.functional as F
from torch import nn


class FeedForwardNetwork(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, kernel, padding, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel[0], padding=padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=kernel[1], padding=padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.layer_norm(x)
        output = output.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output
