import copy
from gentool.ModelBase import ImageModelBase

import numpy as np
from math import floor

import torch
from torch import nn
from torch.cuda import FloatTensor
from torch.autograd import Variable
from torch.optim.adam import Adam
import torch.nn.functional as F
from torchinfo import summary

from gentool.util.ImageToVec import ImageToVec
from gentool.util.VecToImage import VecToImage


class GAN(ImageModelBase):
    def __init__(self, dataloader, image_size, image_channels, layers_per_size, latent_dim, initial_channels=4, learning_rate=1e-4):
        super().__init__(dataloader)

        self.image_size = image_size
        self.image_channels = image_channels
        self.layers_per_size = layers_per_size
        self.latent_dim = latent_dim
        self.initial_channels = initial_channels

        self.generator = VecToImage(image_size,
                                    image_channels,
                                    layers_per_size,
                                    initial_channels=initial_channels,
                                    dense_layers=(latent_dim, latent_dim))

        self.discriminator = ImageToVec(image_size,
                                        image_channels,
                                        layers_per_size,
                                        initial_channels=initial_channels,
                                        dense_layers=(latent_dim, floor(latent_dim / 2), 1),
                                        output_activation=nn.Sigmoid())

        self.optimizer_g = Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=learning_rate)

        summary(self)

    def sample_images(self, count):
        z = self.random_z(count)
        return self.generator(z)

    def train_batch(self, batch):
        # Initialize
        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()

        batch_size = len(batch)
        zeros = Variable(FloatTensor(np.zeros((batch_size, 1))))
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        z_noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
        real_input_noise = Variable(FloatTensor(np.random.normal(0, 0.1, batch.shape)))
        fake_input_noise = Variable(FloatTensor(np.random.normal(0, 0.1, batch.shape)))

        # Update generator
        g_fake_data = self.generator(z_noise) + fake_input_noise

        with torch.no_grad():
            dg_fake_decision = self.discriminator(g_fake_data)
            g_loss = F.mse_loss(dg_fake_decision, ones)

        g_loss.backward()
        self.optimizer_g.step()

        # Update discriminator
        d_real_decision = self.discriminator(batch + real_input_noise)
        d_real_error = F.mse_loss(d_real_decision, ones)

        d_fake_decision = self.discriminator(g_fake_data)
        d_fake_error = F.mse_loss(d_fake_decision, zeros)
        d_loss = d_real_error + d_fake_error

        d_loss.backward()
        self.optimizer_d.step()

        return [g_loss, d_loss]

    def loss_names_and_groups(self):
        return ['g_loss', 'd_loss'], {'GAN': ['g_loss', 'd_loss']}
