import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data.imdb import IMDB
from transformer import Transformer


def get_dataset(dataset_name):
    if dataset_name == "imdb":
        return IMDB("train"), IMDB("test")

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model(args, input_dim):
    return Transformer(
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.d_ff,
        input_dim,
        2,
    )


def train(model, ds, criterion, optimizer, device, batch_size):
    loader = ds.get_loader(batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        text = batch["input_ids"].to(device)
        label = batch["label"].to(device)
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, ds, criterion, device, batch_size):
    loader = ds.get_loader(batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            text = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            output = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def train_loop(
    model, train_ds, test_ds, criterion, optimizer, device, num_epochs, batch_size
):
    for epoch in range(num_epochs):
        train_loss = train(model, train_ds, criterion, optimizer, device, batch_size)
        valid_loss, accuracy = evaluate(model, test_ds, criterion, device, batch_size)
        print(f"Epoch {epoch} train Loss: {train_loss:.3f}")


def main(args):
    ds_train, ds_test = get_dataset(args.dataset)

    input_dim = len(ds_train.vocab)
    m = get_model(args, input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(m.parameters(), lr=args.lr)

    train_loop(
        m, ds_train, ds_test, criterion, optimizer, "cpu", args.epochs, args.batch_size
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train Transformer model")
    arg_parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    arg_parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of model"
    )
    arg_parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    arg_parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of layers"
    )
    arg_parser.add_argument(
        "--d_ff", type=int, default=2048, help="Dimension of feed forward network"
    )
    arg_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    arg_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    arg_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    arg_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = arg_parser.parse_args()

    main(args)
