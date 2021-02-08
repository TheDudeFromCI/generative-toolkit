import os
import numpy as np
from tqdm import tqdm
from time import time
from math import sqrt, log
from abc import abstractmethod

import torch
from torch import nn
from torch.optim.adam import Adam
from torch.cuda import FloatTensor
from torch.nn import functional as F
from torchvision.utils import save_image


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.save_model_rate = 1000
        self.batch_size = 16
        self.gradient_updates = 1

    def save_model(self, update_number):
        os.makedirs('models', exist_ok=True)
        filename = 'models/up_{}.{}.pth.tar'.format(update_number, time())
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)

    def fit(self, n_updates, offset=0):
        with tqdm(range(offset, n_updates + offset), smoothing=0.0) as prog_bar:
            losses = []
            for update_number in range(offset + 1, n_updates + offset + 1):
                losses = self.train_batch()

                prog_bar.write('Batch = {}, Losses = {}'.format(update_number, losses))
                self.batch_callback(update_number, losses)
                prog_bar.update(1)

    def batch_callback(self, update_number, losses):
        if update_number % self.save_model_rate == 0:
            self.save_model(update_number)

    @abstractmethod
    def train_batch(_):
        raise NotImplementedError

    def noise(self, size):
        if len(size) == 1 or len(size) == 3:
            size = (self.batch_size, *size)

        return FloatTensor(np.random.normal(0, 1, size))


class ImageModelBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.save_snapshot_rate = 100

    def batch_callback(self, update_number, losses):
        super().batch_callback(update_number, losses)

        if update_number % self.save_snapshot_rate == 0:
            self.save_snapshot(update_number=update_number)

    def save_snapshot(self, update_number=0):
        os.makedirs('images', exist_ok=True)

        self.train(False)
        with torch.no_grad():
            images, rows = self.sample_images()
        self.train(True)

        filename = 'images/up-{}.{}.png'.format(update_number, time())
        save_image(images, filename, nrow=rows)

    @abstractmethod
    def sample_images(_):
        raise NotImplementedError

    def sample_image_from_noise(self, size):
        z = self.noise(size)
        return self(z), int(sqrt(self.batch_size))

    def sample_image_to_image(self, batch):
        generated = self(batch)
        images = [val for pair in zip(batch, generated) for val in pair]
        return images, int(sqrt(len(batch))) * 2

    def sample_image_to_image_masked(self, batch, modified):
        generated = self(modified)
        images = [val for pair in zip(batch, modified, generated) for val in pair]
        return images, int(sqrt(len(batch))) * 3


class VaeModelBase(ImageModelBase):
    def __init__(self, dataloader, latent_dim, lr=1e-4):
        super().__init__()

        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.kld_weight = 1
        self.logcosh_alpha = 10

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.mu = nn.Linear(latent_dim ** 2, latent_dim)
        self.var = nn.Linear(latent_dim ** 2, latent_dim)

        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.encoder(x)
        mu, var = self.mu(x), self.var(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        x = eps * std + mu

        x = self.decoder(x)
        return x, mu, var

    def vae_logcosh_loss(self, batch):
        generated, mu, var = self(batch)

        t = generated - batch
        recon_loss = self.logcosh_alpha * t + torch.log(1 + torch.exp(-2 * self.logcosh_alpha * t)) - log(2)
        recon_loss = (1 / self.logcosh_alpha) * recon_loss.mean()

        kld_loss = -0.5 * torch.mean(1 + var - mu ** 2 - var.exp()) * self.kld_weight
        return recon_loss + kld_loss

    def sample_images(self):
        images, rows = self.sample_image_to_image(next(self.dataloader))

        noise = self.noise(self.latent_dim)
        images = torch.cat([images, self.decoder(noise)])

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
