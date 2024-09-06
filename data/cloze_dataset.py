import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple
import random
from data.util import build_vocab, encode, tokenize


def _collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets, mask_positions = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    mask_positions = torch.stack(mask_positions)
    return inputs, targets, mask_positions


def _mask_random_word(
    sequence: List[int], vocab: Dict[str, int], mask_token: str
) -> Tuple[List[int], int, int]:
    target_idx = random.randint(0, len(sequence) - 1)
    target_word = sequence[target_idx]
    masked_sequence = sequence[:]
    masked_sequence[target_idx] = vocab.get(mask_token, vocab["<unk>"])
    return masked_sequence, target_word, target_idx


class ClozeDataset(Dataset):
    def __init__(
        self,
        path: str,
        seq_length: int = 128,
        batch_size: int = 32,
        cloze_mask_token: str = "<mask>",
        rotate_n: int = 0,
    ):
        super().__init__()
        self._seq_length = seq_length
        self._batch_size = batch_size
        self._cloze_mask_token = cloze_mask_token
        self._rotate_n = rotate_n
        self._epoch_count = 0

        self._vocab = None

        with open(path, "r") as f:
            self._text = f.read()

        self._words = tokenize(self._text)
        self._encoded_words = encode(self._text, self.vocab)

        self._data = [
            self._encoded_words[i : i + self._seq_length]
            for i in range(len(self._encoded_words) - self._seq_length)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self._data[idx]
        masked_sequence, target_word, target_idx = _mask_random_word(
            sequence, self.vocab, self._cloze_mask_token
        )

        input_ids = torch.tensor(masked_sequence, dtype=torch.long)
        target_id = torch.tensor(target_word, dtype=torch.long)
        mask_pos = torch.tensor(target_idx, dtype=torch.long)

        return input_ids, target_id, mask_pos

    def _rotate_dataset(self) -> None:
        if self._rotate_n > 0:
            self._encoded_words = (
                self._encoded_words[self._rotate_n :]
                + self._encoded_words[: self._rotate_n]
            )
            self._data = [
                self._encoded_words[i : i + self._seq_length]
                for i in range(len(self._encoded_words) - self._seq_length)
            ]

    def get_loader(self, batch_size: int = None, shuffle: bool = False) -> DataLoader:
        batch_size = batch_size or self._batch_size
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn
        )

    def on_epoch_end(self) -> None:
        self._epoch_count += 1
        self._rotate_dataset()

    @property
    def vocab(self) -> Dict[str, List[str]]:
        if self._vocab is None:
            self._vocab = build_vocab({"text": [self._text]})
        return self._vocab

    @property
    def max_len(self) -> int:
        return self._seq_length
