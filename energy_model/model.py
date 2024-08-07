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
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 28 * 28)
        return self.nn(x)
