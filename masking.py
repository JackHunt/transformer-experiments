import torch


def generate_subsequent_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


def generate_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_tgt_mask(tgt, pad_idx):
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_mask = tgt_mask & generate_subsequent_mask(tgt.size(-1))
    return tgt_mask.type_as(tgt_mask.data)
