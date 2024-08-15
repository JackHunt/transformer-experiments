import numpy as np
import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        input_dim,
        output_dim,
        max_len=5000,
        dropout_p=0.1,
    ):
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_ff = d_ff
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._max_len = max_len

        self._embedding = nn.Embedding(input_dim, d_model)
        self._positional_encoding = PositionalEncoding(
            d_model, dropout_p=dropout_p, max_len=max_len
        )

        self._encoder_layers = nn.ModuleList(
            [Encoder(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self._decoder_layers = nn.ModuleList(
            [Decoder(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self._fc = nn.Linear(d_model, output_dim)

    def forward(self, src, src_mask, tgt=None, tgt_mask=None):
        src = self._embedding(src) * np.sqrt(self._embedding.embedding_dim)
        src = self._positional_encoding(src)

        x = src
        for l in self._encoder_layers:
            x = l(x, src_mask)

        for l in self._decoder_layers:
            x = l(x, src_mask, tgt=tgt, tgt_mask=tgt_mask)

        x = self._fc(x)

        return x

    @property
    def d_model(self):
        return self._d_model

    @property
    def n_heads(self):
        return self._n_heads

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def d_ff(self):
        return self._d_ff

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def max_len(self):
        return self._max_len
