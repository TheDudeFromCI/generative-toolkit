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

    def random_z(self, batch_size):
        return Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))

    def d_loop(self, d_real_data, d_gen_input=None):
        self.optimizer_d.zero_grad()

        target = Variable(FloatTensor(np.ones((len(d_real_data), 1))))

        d_real_decision = self.discriminator(d_real_data)
        d_real_error = F.mse_loss(d_real_decision, target)

        if d_gen_input is None:
            d_gen_input = self.random_z(len(d_real_data))

        with torch.no_grad():
            g_fake_data = self.generator(d_gen_input)
            g_fake_data += Variable(FloatTensor(np.random.normal(0, 0.1, g_fake_data.shape)))

        d_fake_decision = self.discriminator(g_fake_data)

        target = Variable(FloatTensor(np.zeros((len(g_fake_data), 1))))

        d_fake_error = F.mse_loss(d_fake_decision, target)

        d_loss = d_real_error + d_fake_error
        d_loss.backward()
        self.optimizer_d.step()

        return d_real_error.item()

    def g_loop(self, d_real_data, unroll_steps):
        self.optimizer_g.zero_grad()
        unroll = unroll_steps > 0

        gen_input = self.random_z(len(d_real_data))

        if unroll:
            backup = copy.deepcopy(self.discriminator.state_dict())
            for _ in range(unroll_steps):
                self.d_loop(d_real_data, d_gen_input=gen_input)

        target = Variable(FloatTensor(np.ones((len(d_real_data), 1))))

        g_fake_data = self.generator(gen_input)
        g_fake_data = g_fake_data + Variable(FloatTensor(np.random.normal(0, 0.1, g_fake_data.shape)))
        dg_fake_decision = self.discriminator(g_fake_data)
        g_error = F.mse_loss(dg_fake_decision, target)

        g_error.backward()
        self.optimizer_g.step()

        if unroll:
            self.discriminator.load_state_dict(backup)
            del backup

        return g_error.item()

    def sample_images(self, count):
        z = self.random_z(count)
        return self.generator(z)

    def train_batch(self, batch):
        d_loss = self.d_loop(batch)
        g_loss = self.g_loop(batch, 0)
        return [g_loss, d_loss]

    def loss_names_and_groups(self):
        return ['g_loss', 'd_loss'], {'GAN': ['g_loss', 'd_loss']}
