from gentool.gan.Discriminator import Discriminator
from gentool.gan.Generator import Generator
from gentool.util.ImageGenerator import ImageGenerator
import numpy as np
from math import log2, ceil, sqrt

import torch
from torch import nn
from torchinfo import summary
from torch.optim.adam import Adam
from torch.cuda import FloatTensor
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T

from gentool.ModelBase import ImageModelBase
from gentool.database.ImageDataLoader import image_dataloader


class Gan2DHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 3
        self.discriminator_layers_per_size = 3
        self.gen_initial_channels = 32
        self.dis_initial_channels = 4
        self.learning_rate = 1e-4
        self.kernel = 5
        self.batch_size = 25
        self.image_folders = []
        self.data_augmentation = T.ToTensor()
        self.data_workers = 4
        self.dropout = 0.4
        self.normalization = 'group'
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.bias_neurons = False
        self.leaky_relu_slope = 0.2
        self.input_noise = 0.1

    @property
    def latent_dim(self):
        return (self.gen_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4

    @property
    def discriminator_out(self):
        return (self.dis_initial_channels << int(log2(self.image_size)))


class Gan2D(ImageModelBase):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        dataloader = image_dataloader(hyper_parameters.image_folders, hyper_parameters.batch_size,
                                      hyper_parameters.data_augmentation, workers=hyper_parameters.data_workers)

        super().__init__(dataloader)
        self.hyper_parameters = hyper_parameters
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        self.optimizer_g = Adam(self.generator.parameters(), lr=hyper_parameters.learning_rate, betas=betas)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=hyper_parameters.learning_rate, betas=betas)

        self.cuda()
        summary(self, (1, hyper_parameters.latent_dim), depth=4)

    def _build_generator(self):
        return Generator(self.hyper_parameters)

    def _build_discriminator(self):
        return Discriminator(self.hyper_parameters)

    def sample_images(self, count):
        z = Variable(FloatTensor(np.random.normal(0, 1, (count, self.hyper_parameters.latent_dim))))
        generated = self.generator(z)

        samples = ceil(count / self.hyper_parameters.batch_size)
        extra = self.hyper_parameters.batch_size * samples - count * samples
        for i in range(samples):
            batch = next(self.dataloader)

            if i == samples - 1 and extra > 0:
                batch = batch[:extra]

            generated = torch.cat([generated, batch])

        return generated, int(sqrt(count) * 2)

    def z_noise(self):
        batch_size = self.hyper_parameters.batch_size
        latent_dim = self.hyper_parameters.latent_dim
        return Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

    def real_label(self):
        batch_size = self.hyper_parameters.batch_size
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        label_noise = Variable(FloatTensor(np.random.normal(0, 0.05, (batch_size, 1))))
        return ones * 0.9 + label_noise

    def fake_label(self):
        batch_size = self.hyper_parameters.batch_size
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        label_noise = Variable(FloatTensor(np.random.normal(0, 0.05, (batch_size, 1))))
        return ones * 0.1 + label_noise

    def train_generator(self, batch):
        self.optimizer_g.zero_grad()

        fake_input_noise = torch.randn_like(batch) * self.hyper_parameters.input_noise
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)

        g_loss = F.binary_cross_entropy(dg_fake_decision, self.real_label())
        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    def train_discriminator_fake(self, batch):
        self.optimizer_d.zero_grad()

        fake_input_noise = torch.randn_like(batch) * self.hyper_parameters.input_noise
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)

        d_loss = F.binary_cross_entropy(dg_fake_decision, self.fake_label())
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_discriminator_real(self, batch):
        self.optimizer_d.zero_grad()

        real_input_noise = torch.randn_like(batch) * self.hyper_parameters.input_noise
        d_real_decision = self.discriminator(batch + real_input_noise)

        d_loss = F.binary_cross_entropy(d_real_decision, self.real_label())
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_batch(self, batch):
        d_loss_real = self.train_discriminator_real(batch)
        d_loss_fake = self.train_discriminator_fake(batch)

        g_loss = self.train_generator(batch)
        d_loss = (d_loss_real + d_loss_fake) / 2

        return [g_loss, d_loss]

    def loss_names(_):
        return 'g_loss', 'd_loss'

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
