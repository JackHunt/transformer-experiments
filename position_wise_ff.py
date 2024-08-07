import torch.nn as nn


class PositionWiseFF(nn.Module):
    def __init__(self, d_model, d_embed, dropout_p=0.1):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Linear(d_model, d_embed),
            nn.Linear(d_embed, d_model),
            nn.Dropout(dropout_p),
            nn.ReLU(),
        )

    def forward(self, x):
        return self._layers(x)
