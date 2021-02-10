import os
import numpy as np
from tqdm import tqdm
from time import time
from math import sqrt, log
from abc import abstractmethod

import torch
import torchinfo
from torch import nn
import torch.autograd as autograd
from torch.optim.adam import Adam
from torch.cuda import FloatTensor
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.autograd.variable import Variable


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

    def noise(_, size):
        return FloatTensor(np.random.normal(0, 1, size))

    def count_params(self):
        parameter_count = {}

        for attribute in dir(self):
            model = getattr(self, attribute)
            if not isinstance(model, nn.Module):
                continue

            parameter_count[attribute] = 0

            for param in model.parameters():
                parameter_count[attribute] += param.numel()

        return parameter_count


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

    def fit(self, n_updates, offset=0):
        self.save_snapshot(update_number=offset)
        super().fit(n_updates, offset)


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


class GanModelBase(ImageModelBase):
    def __init__(self, dataloader, generator, discriminator, latent_dim, lr=1e-4, betas=(0.9, 0.99), summary=False):
        super().__init__()

        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.critic_updates = 5
        self.gradient_penalty_lambda = 10
        self.sample_noise = self.noise((64, self.latent_dim))

        self.generator = generator
        self.discriminator = discriminator

        self.optimizer_g = Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.cuda()

        if summary:
            torchinfo.summary(self, (1, self.latent_dim))

            params = self.count_params()
            print(
                f"Loaded GAN with {params['generator']:,} generator params and {params['discriminator']:,} discriminator params.")

    def forward(self, x):
        return self.discriminator(self.generator(x))

    def sample_images(self):
        images = self.generator(self.sample_noise)
        return images, 8

    def wgan_gp_discriminator_loss(self, batch, noise):
        d_loss_real = self.discriminator(batch).mean()

        generated = self.generator(noise)
        d_loss_fake = self.discriminator(generated).mean()

        gradient_penalty = self.calculate_gradient_penalty(batch.data, generated.data)

        return d_loss_fake - d_loss_real + gradient_penalty

    def wgan_generator_loss(self, noise):
        generated = self.generator(noise)
        return -self.discriminator(generated).mean()

    def calculate_gradient_penalty(self, real_data, fake_data):
        alpha = FloatTensor(np.random.random((real_data.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(FloatTensor(real_data.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * self.gradient_penalty_lambda

    def gan_bce_discriminator_loss(self, batch, noise):
        one = FloatTensor(np.ones((len(batch), 1)))
        zero = FloatTensor(np.zeros((len(batch), 1)))

        real_loss = F.binary_cross_entropy(self.discriminator(batch), one)
        fake_loss = F.binary_cross_entropy(self.discriminator(self.generator(noise)), zero)

        return real_loss + fake_loss

    def gan_bce_generator_loss(self, noise):
        one = FloatTensor(np.ones((len(noise), 1)))
        return F.binary_cross_entropy(self.discriminator(self.generator(noise)), one)

    def random_latent(self):
        return self.noise((self.batch_size, self.latent_dim))

    def train_batch(self):
        g_loss = 0
        d_loss = 0

        for _ in range(self.critic_updates):
            self.optimizer_d.zero_grad()

            for _ in range(self.gradient_updates):
                batch = next(self.dataloader)
                noise = self.random_latent()

                loss = self.wgan_gp_discriminator_loss(batch, noise)
                loss.backward()

                d_loss += loss.item() / self.gradient_updates / self.critic_updates

            self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        for _ in range(self.gradient_updates):
            noise = self.random_latent()

            loss = self.wgan_generator_loss(noise)
            loss.backward()

            g_loss += loss.item() / self.gradient_updates

        self.optimizer_g.step()

        return '[g_loss: {:.6f}, d_loss: {:.6f}]'.format(g_loss, d_loss)
