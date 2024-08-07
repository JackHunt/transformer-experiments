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

    def forward(self, x, mask=None):
        x = self._embedding(x) * np.sqrt(self._embedding.embedding_dim)
        x = self._positional_encoding(x)

        for l in self._encoder_layers:
            x = l(x, mask)

        for l in self._decoder_layers:
            x = l(x, mask)

        x = self._fc(x)

        return x
