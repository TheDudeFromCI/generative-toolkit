import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import flatten


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        channels = encoder.max_channels
        self.mu = nn.Linear(channels, latent_dim)
        self.var = nn.Linear(channels, latent_dim)

    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        var = self.var(x)
        z = self.reparameterize(mu, var)

        z = self.decoder(z)
        return [z, mu, var]

    def reparameterize(_, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def train_batch(self, batch, optimizer):
        optimizer.zero_grad()

        recons, mu, var = self(batch)

        recons_loss = F.binary_cross_entropy(recons, batch, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + var - mu.pow(2) - var.exp())

        loss = recons_loss + kld_loss
        loss.backward()
        optimizer.step()

        return loss.item()

    def sample_output(self, batch):
        with torch.no_grad():
            recons, _, _ = self(batch)

        images = [val for pair in zip(recons, batch) for val in pair]
        return images
