import numpy as np

import torchinfo
import torch.autograd as autograd
from torch.cuda import FloatTensor
from torch.nn import functional as F
from torch.autograd.variable import Variable

from ModelBase import ImageModelBase


class GanModelBase(ImageModelBase):
    def __init__(self, dataloader, generator, discriminator, summary=False):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader

        self.latent_dim = generator.latent_dim
        self.image_size = generator.image_size
        self.image_channels = generator.image_channels
        self.sample_noise = self.noise((64, self.latent_dim))

        self.critic_updates = 5
        self.gradient_penalty_lambda = 10

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

        generated = self.generator(noise).detach()
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
            self.discriminator.optimizer.zero_grad()

            for _ in range(self.gradient_updates):
                batch = next(self.dataloader)
                noise = self.random_latent()

                loss = self.wgan_gp_discriminator_loss(batch, noise)
                loss.backward()

                d_loss += loss.item() / self.gradient_updates / self.critic_updates

            self.discriminator.optimizer.step()

        self.generator.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            noise = self.random_latent()

            loss = self.wgan_generator_loss(noise)
            loss.backward()

            g_loss += loss.item() / self.gradient_updates

        self.generator.optimizer.step()

        return '[g_loss: {:.6f}, d_loss: {:.6f}]'.format(g_loss, d_loss)
