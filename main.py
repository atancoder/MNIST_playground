import os

import click
import torch

from model import CNN
from train import compute_accuracy, train
from utils import get_dataloader, load_saved_model

torch.manual_seed(1337)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
)
print(f"Using {DEVICE} device")


@click.command(name="train_model")
@click.option("--saved_model_file", "saved_model_file", default="model.pt")
def train_model(saved_model_file):
    model = CNN().to(DEVICE)
    train_dataloader = get_dataloader(train=True)
    train(model, train_dataloader, DEVICE)

    torch.save(model.state_dict(), saved_model_file)


@click.command(name="test_model")
@click.option("--model", "saved_model_file", default="model.pt")
def test_model(
    saved_model_file: str,
):
    test_dataloader = get_dataloader(train=False)
    model = load_saved_model(saved_model_file, DEVICE)
    accuracy = compute_accuracy(model, test_dataloader, DEVICE)
    print(f"Accuracy: {accuracy}")


@click.group()
def cli():
    return


cli.add_command(train_model)
cli.add_command(test_model)

if __name__ == "__main__":
    cli()
