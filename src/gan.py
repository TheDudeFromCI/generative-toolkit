import copy
import math
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.cuda import FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, encoder, learning_rate):
        super().__init__()
        self.encoder = encoder

        channels = encoder.max_channels
        half_channels = math.floor(channels/2)
        self.dense = nn.Sequential(
            nn.Linear(channels, half_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(half_channels, 1),
            nn.Sigmoid(),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dense(x)
        return x

    def get_target(_, batch_size):
        return Variable(FloatTensor(np.random.normal(
            0.95, 0.05, (batch_size, 1))))


class Generator(nn.Module):
    def __init__(self, decoder, learning_rate):
        super().__init__()
        self.decoder = decoder

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.decoder(x)
        return x

    def get_target(_, batch_size):
        return Variable(FloatTensor(np.random.normal(
            0.05, 0.05, (batch_size, 1))))


class GAN(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, learning_rate):
        super().__init__()
        self.generator = Generator(decoder, learning_rate)
        self.discriminator = Discriminator(encoder, learning_rate)
        self.latent_dim = latent_dim

    def random_z(self, batch_size):
        return Variable(FloatTensor(np.random.normal(
            0, 1, (batch_size, self.latent_dim))))

    def d_loop(self, d_real_data, d_gen_input=None):
        self.discriminator.optimizer.zero_grad()

        d_real_decision = self.discriminator(d_real_data)
        target = self.discriminator.get_target(len(d_real_data))
        d_real_error = F.binary_cross_entropy(d_real_decision, target)

        if d_gen_input is None:
            d_gen_input = self.random_z(len(d_real_data))

        with torch.no_grad():
            d_fake_data = self.generator(d_gen_input)
        d_fake_decision = self.discriminator(d_fake_data)

        target = self.generator.get_target(len(d_real_data))
        d_fake_error = F.binary_cross_entropy(d_fake_decision, target)

        d_loss = d_real_error + d_fake_error
        d_loss.backward()
        self.discriminator.optimizer.step()

        return d_real_error.item()

    def g_loop(self, d_real_data, unroll_steps):
        self.generator.optimizer.zero_grad()
        unroll = unroll_steps > 0

        gen_input = self.random_z(len(d_real_data))

        if unroll:
            backup = copy.deepcopy(self.discriminator.state_dict())
            for _ in range(unroll_steps):
                self.d_loop(d_real_data, d_gen_input=gen_input)

        g_fake_data = self.generator(gen_input)
        dg_fake_decision = self.discriminator(g_fake_data)
        target = self.generator.get_target(len(d_real_data))
        g_error = F.binary_cross_entropy(dg_fake_decision, target)

        g_error.backward()
        self.generator.optimizer.step()

        if unroll:
            self.discriminator.load_state_dict(backup)
            del backup

        return g_error.item()

    def g_sample(self, count):
        z = self.random_z(count)
        gen_imgs = self.generator(z)
        return gen_imgs.data

    def train_batch(self, batch):
        d_loss = self.d_loop(batch)
        g_loss = self.g_loop(batch, 0)
        return [g_loss, d_loss]
