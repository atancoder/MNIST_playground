import os

import numpy as np
import torch
from datasets import load_dataset, load_metric
from torchvision import datasets, transforms

from cnn_model import CNN


def get_accuracy_metric():
    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    return compute_metrics


def get_images_in_vit_format(processor):
    def process(example_batch):
        batch_images = map(
            transforms.Lambda(lambda x: x.convert("RGB")),
            example_batch["image"],
        )
        inputs = processor([x for x in batch_images], return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs

    mnist = load_dataset("mnist")
    return mnist["train"].with_transform(process), mnist["test"].with_transform(process)

def filter_dataset(dataset, digit_class):
    indices = [i for i, label in enumerate(dataset.targets) if label == digit_class]
    return torch.utils.data.Subset(dataset, indices)

def get_dataloader(train: bool = True, batch_size: int = 64, is_vit: bool = False, digit_class=None):
    processing_transforms = [
        transforms.ToTensor()
    ]  # This automatically scales pixel values to [0, 1]
    if is_vit:
        processing_transforms.append(
            transforms.Resize((224, 224))
        )  # ViT models typically require 224x224 input size
    dataset = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=transforms.Compose(processing_transforms),
    )
    if digit_class is not None:
        dataset = filter_dataset(dataset, digit_class)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_saved_model(model_file, device):
    if not os.path.exists(model_file):
        raise Exception(f"{model_file} not found")
    model = CNN()
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    model.to(device)
    return model
