import torch
from torch import nn

from Encoder import Encoder


class KLEncoder(Encoder):
    def create_network(self, config):
        super().create_network(config)

        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.var = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        x = self.model(x)
        mu, var = self.mu(x), self.var(x)

        std = torch.exp(0.5 * var)
        eps = torch.normal(0, 1, std.shape)
        x = eps * std + mu

        return x, mu, var
