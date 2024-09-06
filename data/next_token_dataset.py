import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict
from data.util import build_vocab, encode


class TinyShakespeare(Dataset):
    def __init__(self, path: str, seq_length: int = 100, batch_size: int = 32):
        super().__init__()
        self._seq_length = seq_length
        self._batch_size = batch_size

        with open(path, "r") as f:
            self._text = f.read()

        self._vocab = None

        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_loader(self, batch_size: int = None, shuffle: bool = False) -> DataLoader:
        raise NotImplementedError

    @property
    def vocab(self) -> Dict[str, List[str]]:
        if self._vocab is None:
            self._vocab = build_vocab({"text": [self._text]})
        return self._vocab

    @property
    def max_len(self) -> int:
        raise NotImplementedError
