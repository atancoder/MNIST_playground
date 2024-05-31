import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, filter_size=3):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, filter_size, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        # Fully connected layer that generates 10 logits
        self.fc = nn.Sequential(
            nn.Linear(16 * 28 * 28, 10),
        )

    def forward(self, input):
        conv_output = self.conv_layer(input)
        flattened = torch.flatten(conv_output, 1)
        output = self.fc(flattened)
        return output
