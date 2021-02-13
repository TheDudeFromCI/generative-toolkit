from math import log

import torch
from torch import nn
from torch.optim.adam import Adam

from gentool.ModelBase import ImageModelBase


class VaeModelBase(ImageModelBase):
    def __init__(self, dataloader, encoder, decoder):
        super().__init__()

        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder

        self.latent_dim = decoder.latent_dim
        self.image_size = decoder.image_size
        self.image_channels = decoder.image_channels

        self.sample_img = next(dataloader)
        self.sample_noise = self.noise((64, self.latent_dim))

        self.kld_weight = 1
        self.logcosh_alpha = 10

        self.cuda()

    def forward(self, x):
        x, mu, var = self.encoder(x)
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
        images, rows = self.sample_image_to_image(self.sample_img)
        images = torch.cat([images, self.decoder(self.sample_noise)])
        return images, rows

    def train_batch(self):
        vae_loss = 0

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            original = next(self.dataloader)
            loss = self.vae_logcosh_loss(original)
            loss.backward()

            vae_loss += loss.item() / self.gradient_updates

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

        return 'loss: {:.6f}'.format(vae_loss)
