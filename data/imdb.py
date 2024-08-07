import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.split()


def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def build_vocab(data, min_freq=1):
    counter = Counter()
    for text in data["text"]:
        counter.update(tokenize(text))

    vocab = {
        word: idx + 2
        for idx, (word, count) in enumerate(counter.items())
        if count >= min_freq
    }

    vocab["<unk>"] = 0
    vocab["<pad>"] = 1

    return vocab


class IMDB(Dataset):
    def __init__(self, mode, max_len=2048):
        assert mode in ["train", "test"]

        self._ds = load_dataset("imdb")[mode]
        self._vocab = None
        self.max_len = max_len

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        text = self._ds[index]["text"]
        label = 1 if self._ds[index]["label"] == "pos" else 0
        encoded_text = encode(text, self.vocab)

        if len(encoded_text) < self.max_len:
            encoded_text.extend(
                [self.vocab["<pad>"]] * (self.max_len - len(encoded_text))
            )
        else:
            encoded_text = encoded_text[: self.max_len]

        return {
            "input_ids": torch.tensor(encoded_text, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = build_vocab(self._ds)
        return self._vocab

    def get_loader(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=True)
