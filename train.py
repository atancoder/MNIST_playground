import time
from typing import List, Optional

import torch
import torch.nn as nn

# Used for multi-class classification
# Handles converting logits to softmax probabilities
criterion = nn.CrossEntropyLoss()


def _get_loss_for_batch(
    model: nn.Module, batch: List[torch.Tensor], device: str
) -> nn.MSELoss:
    data, labels = batch
    logits = model(data.to(device))
    loss = criterion(logits, labels.to(device))
    return loss


def train(model, dataloader, device, epochs=10):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Epoch {epoch} / {epochs}\n")
        for batch in dataloader:
            loss = _get_loss_for_batch(model, batch, device)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Loss Logging
            epoch_loss += loss.item()

        avg_batch_loss = epoch_loss / len(dataloader)
        print(
            f"Epoch complete. Avg batch loss: {avg_batch_loss}. {int((time.time() - start_time) / 60)} minutes have elapsed"
        )
