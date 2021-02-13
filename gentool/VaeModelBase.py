from math import log, sqrt

import torch
import torchinfo

from ModelBase import ImageModelBase


class VaeModelBase(ImageModelBase):
    def __init__(self, dataloader, encoder, decoder, summary=False):
        super().__init__()

        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder

        self.latent_dim = decoder.latent_dim
        self.image_size = decoder.image_size
        self.image_channels = decoder.image_channels

        self.sample_img = next(dataloader)
        self.sample_noise = self.noise((len(self.sample_img), self.latent_dim))

        self.kld_weight = 1
        self.logcosh_alpha = 10

        self.cuda()

        if summary:
            torchinfo.summary(self, (1, self.image_channels, self.image_size, self.image_size))

            params = self.count_params()
            print(
                f"Loaded VAE with {params['encoder']:,} encoder params and {params['decoder']:,} decoder params.")

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
        generated, _, _ = self(self.sample_img)
        images = [val for pair in zip(self.sample_img, generated) for val in pair]
        rows = int(sqrt(len(self.sample_img))) * 3

        images = [*images, *self.decoder(self.sample_noise)]
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
