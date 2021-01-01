import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mu, var = self.encoder(x)
        return [self.decoder(z), mu, var]

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
