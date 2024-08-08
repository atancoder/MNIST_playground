import os
import sys

import click
import torch

# Get the parent directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "..")
sys.path.append(parent_dir)
from model import VAE
from torchvision import transforms

from utils import get_dataloader

torch.manual_seed(1337)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore
)
print(f"Using {DEVICE} device")
LR = 1e-3
EPOCHS = 50
Z_DIM = 96


def gen_image(model):
    z = torch.randn_like(torch.rand(Z_DIM)).to(DEVICE).unsqueeze(0)
    return model.decode(z)


def validate(model, dataloader):
    total_loss = 0
    for _, batch in enumerate(dataloader):
        data = batch[0].to(DEVICE)
        recon_data, mu, logvar = model(data)
        recon_loss = torch.nn.functional.binary_cross_entropy(
            recon_data, data, reduction="sum"
        )
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_divergence

        total_loss += loss.item()
    avg_batch_loss = total_loss / len(dataloader)
    print(f"Validation batch loss: {avg_batch_loss}")


def train(model, dataloader, validation_dataloader, saved_model_file):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()
    for epoch in range(EPOCHS):
        print(f"Starting epoch: {epoch}")
        total_loss = 0
        for batch_id, batch in enumerate(dataloader):
            data = batch[0].to(DEVICE)
            recon_data, mu, logvar = model(data)

            recon_loss = torch.nn.functional.binary_cross_entropy(
                recon_data, data, reduction="sum"
            )
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} avg batch loss: {total_loss / len(dataloader)}")
        # validate(model, validation_dataloader)
        torch.save(model.state_dict(), saved_model_file)
        print("Saved model")


def load_saved_model(model_file):
    print("Loading saved model")
    model = VAE(z_dim=Z_DIM, device=DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    return model


@click.command(name="train_model")
@click.option("--saved_model_file", "saved_model_file", default="model.pt")
def train_model(saved_model_file):
    if os.path.exists(saved_model_file):
        model = load_saved_model(saved_model_file)
    else:
        model = VAE(z_dim=Z_DIM, device=DEVICE).to(DEVICE)
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    train(model, train_dataloader, test_dataloader, saved_model_file)


@click.command(name="gen_images")
@click.option("--model", "saved_model_file", default="model.pt")
@click.option("--output_dir", default="images/")
def gen_images(
    saved_model_file: str,
    output_dir: str,
):
    num_images = 5
    model = load_saved_model(saved_model_file)
    to_pil = transforms.ToPILImage()
    for i in range(num_images):
        image = gen_image(model)
        pil_image = to_pil(image.squeeze(1))  # Remove the channel dimension
        pil_image.save(os.path.join(output_dir, f"image_{i}.png"))


@click.group()
def cli():
    return


cli.add_command(train_model)
cli.add_command(gen_images)

if __name__ == "__main__":
    cli()
