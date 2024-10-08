import numpy as np
import torch
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.positional_encoding import PositionalEncoding

import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        input_dim: int,
        output_dim: int,
        max_len: int = 5000,
        dropout_p: float = 0.1,
        decoder_only: bool = False,
    ):
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_ff = d_ff
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._max_len = max_len
        self._decoder_only = decoder_only

        self._embed = nn.Embedding(input_dim, d_model)
        self._pos = PositionalEncoding(d_model, dropout_p=dropout_p, max_len=max_len)

        if not self.decoder_only:
            self._embed_enc = nn.Embedding(input_dim, d_model)
            self._pos_enc = PositionalEncoding(
                d_model, dropout_p=dropout_p, max_len=max_len
            )
            self._encoder_layers = nn.ModuleList(
                [Encoder(d_model, n_heads, d_ff) for _ in range(n_layers)]
            )

        self._decoder_layers = nn.ModuleList(
            [Decoder(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self._fc = nn.Linear(d_model, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_enc: torch.Tensor = None,
        x_enc_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if x_enc is not None and self.decoder_only:
            raise ValueError("x_enc provided but decoder_only is True")

        if not self.decoder_only:
            if x_enc is None:
                x_enc = x  # TODO: check if this is correct

            x_enc = self._embed_enc(x_enc) * np.sqrt(self._embed_enc.embedding_dim)
            x_enc = self._pos_enc(x_enc)
            for l in self._encoder_layers:
                x_enc = l(x_enc, x_mask)

        x = self._embed(x) * np.sqrt(self._embed.embedding_dim)
        x = self._pos(x)
        for l in self._decoder_layers:
            x = l(x, x_mask, x_enc=x_enc, x_enc_mask=x_enc_mask)

        return self._fc(x)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def d_ff(self) -> int:
        return self._d_ff

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def decoder_only(self) -> bool:
        return self._decoder_only
