import argparse


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


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
