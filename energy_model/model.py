from torch import nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 28 * 28  # 784
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, 49),
            nn.ReLU(),
            nn.Linear(49, 1),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 28 * 28)
        output = self.nn(x)
        return output


class EnergyCNNModel(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        input_dim = 28 * 28  # 784
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),  
            # Start Compressing
            nn.Conv2d(32, 32, kernel_size=kernel_size, stride=2),  # new dimension = ((28 - 3) / 2) + 1 = 13 X 13
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=kernel_size, stride=2),  # new dimension = ((13 - 3) / 2) + 1 = 6 X 6
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.final_linear = nn.Linear(32*6*6, 1)
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

    def forward(self, x):
        """
        Shape = B X 1 X 28 X 28
        """
        batch_size = x.shape[0]
        conv_layer_output = self.conv_layers(x)  # B X 32 X 6 X 6
        flattened = conv_layer_output.view(batch_size, 32*6*6)
        return self.final_linear(flattened)
