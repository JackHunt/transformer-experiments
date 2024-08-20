import torch
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_ff import PositionWiseFF
from torch import Tensor

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float = 0.1):
        super().__init__()

        self._dropout = nn.Dropout(dropout_p)

        self._attn = MultiHeadAttention(d_model, n_heads)
        self._norm1 = nn.LayerNorm(d_model)

        self._ff = PositionWiseFF(d_model, d_ff, dropout_p)
        self._norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attn_out, _ = self._attn(x, x, x, mask)
        x = x + self._dropout(attn_out)
        x = self._norm1(x)

        ff_out = self._ff(x)
        x = x + self._dropout(ff_out)
        x = self._norm2(x)

        return x
