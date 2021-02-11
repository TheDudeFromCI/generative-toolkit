from math import log

import torch
from torch import nn
from torch.optim.adam import Adam

from gentool.ModelBase import ImageModelBase


class VaeModelBase(ImageModelBase):
    def __init__(self, dataloader, encoder, decoder, latent_dim, lr=1e-4):
        super().__init__()

        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.kld_weight = 1
        self.logcosh_alpha = 10
        self.sample_images = next(dataloader)
        self.sample_noise = self.random_latent()

        self.encoder = encoder
        self.decoder = decoder

        self.mu = nn.Linear(latent_dim ** 2, latent_dim)
        self.var = nn.Linear(latent_dim ** 2, latent_dim)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.cuda()

    def forward(self, x):
        x = self.encoder(x)
        mu, var = self.mu(x), self.var(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        x = eps * std + mu

        x = self.decoder(x)
        return x, mu, var

    def random_latent(self):
        return self.noise((self.latent_dim,))

    def vae_logcosh_loss(self, batch):
        generated, mu, var = self(batch)

        t = generated - batch
        recon_loss = self.logcosh_alpha * t + torch.log(1 + torch.exp(-2 * self.logcosh_alpha * t)) - log(2)
        recon_loss = (1 / self.logcosh_alpha) * recon_loss.mean()

        kld_loss = -0.5 * torch.mean(1 + var - mu ** 2 - var.exp()) * self.kld_weight
        return recon_loss + kld_loss

    def sample_images(self):
        images, rows = self.sample_image_to_image(self.sample_images)
        images = torch.cat([images, self.decoder(self.sample_noise)])
        return images, rows

    def train_batch(self):
        vae_loss = 0

        self.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            original = next(self.dataloader)
            loss = self.vae_logcosh_loss(original)
            loss.backward()

            vae_loss += loss.item() / self.gradient_updates

        self.optimizer.step()

        return 'loss: {:.6f}'.format(vae_loss)
