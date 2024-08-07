import os
import sys

import click
import torch

# Get the parent directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.append(parent_dir)
from model import EnergyModel
from torchvision import transforms

from utils import get_dataloader

torch.manual_seed(1337)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
)
print(f"Using {DEVICE} device")
LR = 1e-5
EPOCHS = 10


def gen_fake_data(model, data, n_steps=10, lr=1e-3, noise_scale=0.005):
    """
    Langevin Dynamics MCMC
    Start with random noise, then iteratively improve noise by
    doing gradient descent based on energy function

    Note: we don't want to update the model's weights, only the fake_data's
    """
    fake_data = torch.randn_like(data)
    fake_data.requires_grad = True
    for _ in range(n_steps):
        energy = model(fake_data).mean()
        energy.backward()
        # find gradient for fake_data and update it based on learning rate
        fake_data.data -= lr * fake_data.grad
        fake_data.data = torch.clip(fake_data.data, 0, 1)
        fake_data.grad.zero_()

    fake_data.requires_grad = False
    return fake_data


def train(model, dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()
    for epoch in range(EPOCHS):
        print(f"Starting epoch: {epoch}")
        for batch_id, batch in enumerate(dataloader):
            data = batch[0].to(DEVICE)
            energy_score = model(data).mean()
            fake_energy_score = model(gen_fake_data(model, data)).mean()

            cd_loss = energy_score - fake_energy_score
            optimizer.zero_grad()  # zero grad here before step since generating fake data may have accumulated gradients
            cd_loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                print(f"Batch {batch_id} loss: {cd_loss.item()}")
                print(f"real energy: {energy_score}. fake energy: {fake_energy_score}")


def load_saved_model(model_file):
    model = EnergyModel()
    model.load_state_dict(torch.load(model_file, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    return model


@click.command(name="train_model")
@click.option("--saved_model_file", "saved_model_file", default="model.pt")
def train_model(saved_model_file):
    model = EnergyModel().to(DEVICE)
    train_dataloader = get_dataloader(train=True)
    train(model, train_dataloader)
    torch.save(model.state_dict(), saved_model_file)
    print("Saved model")


@click.command(name="gen_images")
@click.option("--model", "saved_model_file", default="model.pt")
@click.option("--output_dir", default="images/")
def gen_images(
    saved_model_file: str,
    output_dir: str,
):
    model = load_saved_model(saved_model_file)
    data = torch.rand(10, 1, 28, 28).to(DEVICE)
    fake_data = gen_fake_data(model, data, n_steps=1000)
    fake_data_energy = model(fake_data).mean()
    print(f"Fake energy: {fake_data_energy.item()}")

    # train_dataloader = get_dataloader(train=True)
    # fake_data = next(iter(train_dataloader))[0]
    to_pil = transforms.ToPILImage()
    for i in range(10):
        pil_image = to_pil(fake_data[i].squeeze(0))  # Remove the channel dimension
        pil_image.save(os.path.join(output_dir, f"image_{i}.png"))


@click.group()
def cli():
    return


cli.add_command(train_model)
cli.add_command(gen_images)

if __name__ == "__main__":
    cli()
