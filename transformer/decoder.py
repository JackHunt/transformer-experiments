import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_wise_ff import PositionWiseFF


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0.1):
        super().__init__()

        self._dropout = nn.Dropout(dropout_p)

        self._self_attn = MultiHeadAttention(d_model, n_heads)
        self._norm1 = nn.LayerNorm(d_model)

        self._cross_attn = MultiHeadAttention(d_model, n_heads)
        self._norm2 = nn.LayerNorm(d_model)

        self._ff = PositionWiseFF(d_model, d_ff)
        self._norm3 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask, tgt=None, tgt_mask=None):
        if tgt_mask is not None and tgt is None:
            raise ValueError("tgt_mask provided but tgt is None")

        x = src
        attn_out, _ = self._self_attn(x, x, x, tgt_mask)
        x = x + self._dropout(attn_out)
        x = self._norm1(x)

        if tgt is not None:
            attn_out, _ = self._cross_attn(x, tgt, tgt, src_mask)
            x = x + self._dropout(attn_out)
            x = self._norm2(x)

        ff_out = self._ff(x)
        x = x + self._dropout(ff_out)
        return self._norm3(x)
