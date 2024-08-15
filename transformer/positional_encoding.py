import torch
import torch.nn as nn

import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout_p=0.1, max_len=5000):
        super().__init__()
        self._dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self._dropout(x)
