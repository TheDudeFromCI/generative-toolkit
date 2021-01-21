import numpy as np
from math import ceil, sqrt

import torch
from torchinfo import summary
from torch.cuda import FloatTensor
from torch.autograd import Variable
from torch.nn import functional as F

from gentool.ModelBase import ImageModelBase
from gentool.gan.Generator import Generator
from gentool.gan.Discriminator import Discriminator
from gentool.gan.HyperParameters import Gan2DHyperParameters
from gentool.database.ImageDataLoader import image_dataloader


class Gan2D(ImageModelBase):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        dataloader = image_dataloader(hyper_parameters.image_folders, hyper_parameters.batch_size,
                                      hyper_parameters.data_augmentation, workers=hyper_parameters.data_workers)

        super().__init__(dataloader)
        self.hyper_parameters = hyper_parameters
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

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
        s = self.hyper_parameters.label_smoothing

        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        return ones * (1-s)

    def fake_label(self):
        batch_size = self.hyper_parameters.batch_size
        s = self.hyper_parameters.label_smoothing

        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        return ones * s

    def train_generator(self):
        self.generator.optimizer.zero_grad()

        batch_size = self.hyper_parameters.batch_size
        c, s = self.hyper_parameters.image_channels, self.hyper_parameters.image_size

        fake_input_noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, c, s, s))))
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise * self.hyper_parameters.input_noise
        dg_fake_decision = self.discriminator(g_fake_data)

        g_loss = F.binary_cross_entropy(dg_fake_decision, self.real_label())
        g_loss.backward()
        self.generator.optimizer.step()

        return g_loss.item()

    def train_discriminator_fake(self, batch):
        self.discriminator.optimizer.zero_grad()

        fake_input_noise = torch.randn_like(batch) * self.hyper_parameters.input_noise
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)

        d_loss = F.binary_cross_entropy(dg_fake_decision, self.fake_label())
        d_loss.backward()
        self.discriminator.optimizer.step()

        return d_loss.item()

    def train_discriminator_real(self, batch):
        self.discriminator.optimizer.zero_grad()

        real_input_noise = torch.randn_like(batch) * self.hyper_parameters.input_noise
        d_real_decision = self.discriminator(batch + real_input_noise)

        d_loss = F.binary_cross_entropy(d_real_decision, self.real_label())
        d_loss.backward()
        self.discriminator.optimizer.step()

        return d_loss.item()

    def train_batch(self):
        batch = next(self.dataloader)

        d_loss_real = self.train_discriminator_real(batch)
        d_loss_fake = self.train_discriminator_fake(batch)
        g_loss = self.train_generator()

        return '[g_loss: {:.4f}, d_loss_real: {:.4f}, d_loss_fake: {:.4f}]'.format(g_loss, d_loss_real, d_loss_fake)

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
