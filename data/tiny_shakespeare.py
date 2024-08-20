import torch
from torch.utils.data import DataLoader, Dataset

from data.util import build_vocab, encode


class TinyShakespeare(Dataset):
    def __init__(self, path, seq_length=100, batch_size=32):
        super().__init__()
        self._seq_length = seq_length
        self._batch_size = batch_size

        with open(path, "r") as f:
            self._text = f.read()

        self._vocab = None

        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_loader(self, batch_size=None, shuffle=False):
        raise NotImplementedError

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = build_vocab({"text": [self._text]})
        return self._vocab

    @property
    def max_len(self):
        raise NotImplementedError
