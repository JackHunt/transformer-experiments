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

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_out = self._self_attn(x, x, x, tgt_mask)
        x = x + self._dropout(attn_out)
        x = self._norm1(x)

        attn_out = self._cross_attn(x, enc_out, enc_out, src_mask)
        x = x + self._dropout(attn_out)
        x = self._norm2(x)

        ff_out = self._ff(x)
        x = x + self._dropout(ff_out)
        return self._norm3(x)
