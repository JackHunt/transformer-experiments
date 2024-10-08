import argparse
from typing import Tuple
import torch
from data.imdb import IMDB
from transformer.masking import generate_x_mask, generate_x_enc_mask
from transformer.transformer import Transformer

import torch.nn as nn
import torch.optim as optim


def get_dataset(dataset_name: str) -> Tuple[IMDB, IMDB]:
    if dataset_name == "imdb":
        return IMDB("train"), IMDB("test")

    raise ValueError(f"Unknown dataset: {dataset_name}")


class SequenceClassifier(nn.Module):
    def __init__(self, transformer: Transformer, num_classes: int = 2):
        super().__init__()

        self._transformer = transformer

        n = num_classes if num_classes > 2 else 1
        act_type = nn.Softmax if n > 2 else nn.Sigmoid
        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._transformer.output_dim * self._transformer.max_len, n),
            act_type(),
        )

        print(
            "Building a transformer based classifier with the following configuration:\n"
            f"-> d_model: {self._transformer.d_model}\n"
            f"-> n_heads: {self._transformer.n_heads}\n"
            f"-> n_layers: {self._transformer.n_layers}\n"
            f"-> d_ff: {self._transformer.d_ff}\n"
            f"-> input_dim: {self._transformer.input_dim}\n"
            f"-> output_dim: {self._transformer.output_dim}\n"
            f"-> max_len: {self._transformer.max_len}\n"
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self._transformer(src, src_mask)
        return torch.squeeze(self._classifier(x))


def get_model(
    args: argparse.Namespace, input_dim: int, max_len: int
) -> SequenceClassifier:
    return SequenceClassifier(
        Transformer(
            args.d_model,
            args.num_heads,
            args.num_layers,
            args.d_ff,
            input_dim,
            32,
            max_len=max_len,
        )
    )


def train(
    model: SequenceClassifier,
    ds: IMDB,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    batch_size: int,
    max_batches: int = None,
) -> float:
    loader = ds.get_loader(batch_size=batch_size, shuffle=True)

    src_mask = None

    model.train()
    total_loss = 0
    for n, batch in enumerate(loader):
        if max_batches is not None and n >= max_batches:
            return total_loss / (n + 1)

        optimizer.zero_grad()

        text = batch["input_ids"].to(device)
        label = batch["label"].to(device)

        if src_mask is None:
            src_mask = generate_x_mask(text, 2)

        output = model(text, src_mask)

        loss = criterion(output, label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: SequenceClassifier,
    ds: IMDB,
    criterion: nn.Module,
    device: str,
    batch_size: int,
) -> Tuple[float, float]:
    loader = ds.get_loader(batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            text = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            output = model(text, label)

            loss = criterion(output, label)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)


def train_loop(
    model: SequenceClassifier,
    train_ds: IMDB,
    test_ds: IMDB,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int,
    batch_size: int,
    max_batches: int = None,
) -> None:
    for epoch in range(num_epochs):
        train_loss = train(
            model,
            train_ds,
            criterion,
            optimizer,
            device,
            batch_size,
            max_batches=max_batches,
        )

        valid_str = ""
        if not args.skip_validation:
            valid_loss, accuracy = evaluate(
                model, test_ds, criterion, device, batch_size
            )
            valid_str += f"\n\ttest loss: {valid_loss} \n\ttest accuracy: {accuracy}"

        print(f"Epoch {epoch}\n\ttrain loss: {train_loss}{valid_str}")


def main(args: argparse.Namespace) -> None:
    ds_train, ds_test = get_dataset(args.dataset)

    input_dim = len(ds_train.vocab)
    m = get_model(args, input_dim, ds_train.max_len)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(m.parameters(), lr=args.lr)

    train_loop(
        m,
        ds_train,
        ds_test,
        criterion,
        optimizer,
        "cpu",
        args.epochs,
        args.batch_size,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train Transformer model")
    arg_parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    arg_parser.add_argument(
        "--d_model", type=int, default=128, help="Dimension of model"
    )
    arg_parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    arg_parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of layers"
    )
    arg_parser.add_argument(
        "--d_ff", type=int, default=128, help="Dimension of feed forward network"
    )
    arg_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    arg_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    arg_parser.add_argument("--skip_validation", action="store_true")
    arg_parser.add_argument("--max_batches", type=int, default=None)

    args = arg_parser.parse_args()

    main(args)
