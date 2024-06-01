import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input) + input


class CNN(nn.Module):
    def __init__(self, filter_size=3):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, filter_size, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, filter_size, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # new dimension = ((28 - 3) / 2) + 1 = 13 X 13
        conv_layer_2 = nn.Sequential(
            nn.Conv2d(32, 32, filter_size, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.residual = Residual(conv_layer_2)

        # Fully connected layer that generates 10 logits
        self.fc = nn.Sequential(
            nn.Linear(32 * 13 * 13, 10),
        )

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of Linear layers using He initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.residual(output)
        flattened = torch.flatten(output, 1)
        output = self.fc(flattened)
        return output
