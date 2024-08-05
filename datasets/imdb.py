import torch

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class IMDBDataset:
    def __init__(self, split=("train", "test")):
        self._train_iter, self.test_iter = IMDB(split=split)
        self._tokenizer = get_tokenizer("basic_english")
        self._vocab = self._build_vocab()

    def _build_vocab(self):
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield self._tokenizer(text)

        vocab = build_vocab_from_iterator(
            yield_tokens(self._train_iter), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])

        return vocab

    def _data_process(self, raw_text_iter):
        return [
            torch.tensor(self._vocab(self._tokenizer(item[1])), dtype=torch.long)
            for item in raw_text_iter
        ]

    def _collate_batch(self, batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(1 if _label == "pos" else 0)
            processed_text = torch.tensor(
                self._vocab(self._tokenizer(_text)), dtype=torch.int64
            )
            text_list.append(processed_text)
            lengths.append(len(processed_text))

        label_list = torch.tensor(label_list, dtype=torch.float32)
        text_list = pad_sequence(text_list, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.int64)

        return text_list, label_list

    def get_loader(self, batch_size=32, shuffle=True):
        data = self._data_process(self._train_iter)
        return DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_batch
        )
