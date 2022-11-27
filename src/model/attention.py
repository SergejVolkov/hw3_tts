import torch
import numpy as np
import math

from torch import nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask = None):
        # q, k, v: [ batch_size x n_heads x seq_len x hidden_size ]

        attn = torch.matmul(q, torch.transpose(k, -1, -2))
        attn /= self.temperature

        # attn: [ batch_size x n_heads x seq_len x seq_len ]

        if mask is not None:
            attn = torch.masked_fill(attn, mask, -math.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        # output: [ batch_size x n_heads x seq_len x hidden_size ]

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout = 0.1):
        super().__init__()

        assert d_k == d_v

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.c_attn = nn.Linear(d_model, 3 * n_head * d_k)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.c_attn.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + 4 * self.d_k)))

    def forward(self, x, mask = None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_x, _ = x.size()

        residual = x
        x = self.layer_norm(x)

        q, k, v = self.c_attn(x).split(n_head * d_k, dim=-1)
        q = q.view(sz_b, len_x, n_head, d_k).transpose(1, 2)  # b x n x lq x dk
        k = k.view(sz_b, len_x, n_head, d_k).transpose(1, 2)  # b x n x lk x dk
        v = v.view(sz_b, len_x, n_head, d_v).transpose(1, 2)  # b x n x lv x dv

        if mask is not None:
            mask = mask.unsqueeze(1)  # b x 1 x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual

        return output, attn
