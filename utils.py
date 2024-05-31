import os

import torch
from torchvision import datasets, transforms

from model import CNN

# Define a transform to min-max normalize the data
TRANSFORM = transforms.Compose(
    [transforms.ToTensor()]  # This automatically scales pixel values to [0, 1]
)


def get_dataloader(train: bool = True, batch_size: int = 64):
    dataset = datasets.MNIST(
        root="data", train=train, download=True, transform=TRANSFORM
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_saved_model(model_file, device):
    if not os.path.exists(model_file):
        raise Exception(f"{model_file} not found")
    model = CNN()
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    model.to(device)
    return model


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for data, label in dataloader:
            batch_size = data.size(0)
            logits = model(data.to(device))
            probabilities = torch.softmax(logits, 1)
            # choose highest probability index
            predicted = torch.max(probabilities, 1)[1]
            total += batch_size
            correct += (predicted == label.to(device)).sum().item()
    return correct / total
