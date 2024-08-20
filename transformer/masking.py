import torch
from torch import Tensor


def generate_subsequent_mask(seq_len: int) -> Tensor:
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


def generate_x_mask(x: Tensor, pad_idx: int) -> Tensor:
    return (x != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_x_enc_mask(x: Tensor, pad_idx: int) -> Tensor:
    x_enc_mask = (x != pad_idx).unsqueeze(-2)
    x_enc_mask = x_enc_mask & generate_subsequent_mask(x.size(-1))
    return x_enc_mask.type_as(x_enc_mask.data)
