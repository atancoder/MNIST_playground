from typing import Tuple

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, z_dim, device):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
        )
        self.hidden_to_mu = nn.Linear(196, z_dim)
        self.hidden_to_logvar = nn.Linear(196, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 196),
            nn.ReLU(),
            nn.Linear(196, 392),
            nn.ReLU(),
            nn.Linear(392, 28 * 28),
            nn.Sigmoid(),  # Ensures outputs are in range [0,1]
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def encode(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 28 * 28)
        h = self.encoder(x)
        mu = self.hidden_to_mu(h)
        logvar = self.hidden_to_logvar(h)
        return mu, logvar

    def decode(self, z):
        batch_size = z.shape[0]
        decoded_image = self.decoder(z)
        decoded_image = decoded_image.view(batch_size, 1, 28, 28)
        return decoded_image

    def sample_z(self, batch_size, mu, logvar):
        stdev = torch.exp(logvar) ** 0.5
        epsilon = torch.randn_like(torch.rand(batch_size, self.z_dim)).to(self.device)
        z = mu + epsilon * stdev
        return z

    def forward(self, x) -> Tuple[torch.Tensor, float, float]:
        """
        X.shape = B X 1 X 28 X 28
        """
        mu, logvar = self.encode(x)
        z = self.sample_z(x.shape[0], mu, logvar)
        decoded_image = self.decode(z)
        return decoded_image, mu, logvar
