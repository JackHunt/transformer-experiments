import torch

import torch.nn as nn


class PositionWiseFF(nn.Module):
    def __init__(self, d_model: int, d_embed: int, dropout_p: float = 0.1):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Linear(d_model, d_embed),
            nn.Linear(d_embed, d_model),
            nn.Dropout(dropout_p),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)
