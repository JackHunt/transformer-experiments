import torch.nn as nn

from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_ff import PositionWiseFF


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

    def forward(self, x, x_mask, x_enc=None, x_enc_mask=None):
        if x_enc_mask is not None and x_enc is None:
            raise ValueError("x_enc_mask provided but x_enc is None")

        attn_out, _ = self._self_attn(x, x, x, x_mask)
        x = x + self._dropout(attn_out)
        x = self._norm1(x)

        if x_enc is not None:
            attn_out, _ = self._cross_attn(x, x_enc, x_enc, x_enc_mask)
            x = x + self._dropout(attn_out)
            x = self._norm2(x)

        ff_out = self._ff(x)
        x = x + self._dropout(ff_out)
        return self._norm3(x)
