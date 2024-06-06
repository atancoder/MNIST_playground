import os

import click
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from model import CNN
from train import compute_accuracy, train
from utils import (
    get_accuracy_metric,
    get_dataloader,
    get_images_in_vit_format,
    load_saved_model,
)

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


@click.command(name="fine_tune")
@click.option("--saved_model")
def fine_tune(saved_model):
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    train_data, test_data = get_images_in_vit_format(processor)
    if saved_model:
        print("Loading saved model")
        model = ViTForImageClassification.from_pretrained(saved_model)
        print("fin")
    else:
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=get_accuracy_metric(),
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=processor,
    )
    if not saved_model:
        print("Training the model")
        trainer.train()

    print("Evaluating the model")
    print(trainer.evaluate())


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
cli.add_command(fine_tune)
cli.add_command(test_model)

if __name__ == "__main__":
    cli()
