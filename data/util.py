import re
from collections import Counter
from typing import List, Dict


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text.split()


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def build_vocab(data: Dict[str, List[str]], min_freq: int = 1) -> Dict[str, int]:
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
