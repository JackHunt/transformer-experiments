import numpy as np

import torch
import torch.nn as nn


def scaled_dot_product(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d, n):
        super().__init__()

        self._d = d
        self._n = n
        self._d_k = self._d // self._n

        self._W_Q = nn.Linear(self._d, self._d)
        self._W_K = nn.Linear(self._d, self._d)
        self._W_V = nn.Linear(self._d, self._d)
        self._W_O = nn.Linear(self._d, self._d)

        self.attn_fn = scaled_dot_product

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0)

        def split(X):
            return X.view(bs, -1, self._n, self._d_k).transpose(1, 2)

        def combine(X):
            return X.transpose(1, 2).contiguous().view(bs, -1, self._d)

        Q = split(self._W_Q(Q))
        K = split(self._W_K(K))
        V = split(self._W_V(V))

        out, attn = self.attn_fn(Q, K, V, mask=mask)

        out = self._W_O(combine(out))

        return out, attn
