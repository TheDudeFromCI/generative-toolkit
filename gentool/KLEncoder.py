import torch
from torch import nn

from Encoder import Encoder


class KLEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)

        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.var = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        x = self.model(x)
        mu, var = self.mu(x), self.var(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        x = eps * std + mu

        return x, mu, var
