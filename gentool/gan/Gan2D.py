import numpy as np
from math import log2

import torch
from torch import nn
from torchinfo import summary
from torch.optim.adam import Adam
from torch.cuda import FloatTensor
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T

from gentool.ModelBase import ImageModelBase
from gentool.util.ImageToVec import ImageToVec
from gentool.util.VecToImage import VecToImage
from gentool.database.ImageDataLoader import image_dataloader


class Gan2DHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 3
        self.generator_layers_per_size = 3
        self.discriminator_layers_per_size = 3
        self.initial_channels = 4
        self.learning_rate = 1e-4
        self.kernel = 5
        self.batch_size = 25
        self.image_folders = []
        self.data_augmentation = T.ToTensor()
        self.data_workers = 4
        self.dropout = 0.4
        self.normalization = 'group'

    @property
    def latent_dim(self):
        return (self.initial_channels << int(log2(self.image_size) - 2)) * 4 * 4


class Gan2D(ImageModelBase):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        dataloader = image_dataloader(hyper_parameters.image_folders, hyper_parameters.batch_size,
                                      hyper_parameters.data_augmentation, workers=hyper_parameters.data_workers)
        super().__init__(dataloader)
        self.hyper_parameters = hyper_parameters
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.dense = self._build_dense()

        self.optimizer_g = Adam(self.generator.parameters(), lr=hyper_parameters.learning_rate)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=hyper_parameters.learning_rate)

        self.cuda()
        summary(self, (1, hyper_parameters.latent_dim), depth=3)

    def _build_generator(self):
        return VecToImage(self.hyper_parameters.image_size,
                          self.hyper_parameters.image_channels,
                          self.hyper_parameters.generator_layers_per_size,
                          initial_channels=self.hyper_parameters.initial_channels,
                          kernel=self.hyper_parameters.kernel,
                          dropout=self.hyper_parameters.dropout,
                          normalization=self.hyper_parameters.normalization,
                          min_size=4)

    def _build_discriminator(self):
        return ImageToVec(self.hyper_parameters.image_size,
                          self.hyper_parameters.image_channels,
                          self.hyper_parameters.discriminator_layers_per_size,
                          initial_channels=self.hyper_parameters.initial_channels,
                          kernel=self.hyper_parameters.kernel,
                          dropout=self.hyper_parameters.dropout,
                          normalization=self.hyper_parameters.normalization,
                          min_size=4,
                          output_activation=nn.LeakyReLU(inplace=True),
                          normalize_last=True)

    def _build_dense(self):
        return nn.Sequential(
            nn.Linear(self.hyper_parameters.latent_dim, 1),
            nn.Tanh()
        )

    def sample_images(self, count):
        z = Variable(FloatTensor(np.random.normal(0, 1, (count, self.hyper_parameters.latent_dim))))
        return self.generator(z)

    def z_noise(self):
        batch_size = self.hyper_parameters.batch_size
        latent_dim = self.hyper_parameters.latent_dim
        return Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

    def real_label(self):
        batch_size = self.hyper_parameters.batch_size
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        label_noise = Variable(FloatTensor(np.random.normal(0, 0.2, (batch_size, 1))))
        return ones * 0.8 + label_noise

    def fake_label(self):
        batch_size = self.hyper_parameters.batch_size
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        label_noise = Variable(FloatTensor(np.random.normal(0, 0.2, (batch_size, 1))))
        return ones * -0.8 + label_noise

    def train_generator(self, batch):
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        fake_input_noise = torch.randn_like(batch)
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise * 0.2
        dg_fake_decision = self.discriminator(g_fake_data)
        dg_fake_decision = self.dense(dg_fake_decision)

        g_loss = F.mse_loss(dg_fake_decision, self.real_label())
        g_loss.backward()
        self.optimizer_g.step()

        d_loss = F.mse_loss(dg_fake_decision, self.fake_label())
        d_loss.backward()
        self.optimizer_d.step()

        return g_loss.item(), d_loss.item()

    def train_discriminator_real(self, batch):
        self.optimizer_d.zero_grad()

        d_real_decision = self.discriminator(batch + torch.randn_like(batch) * 0.2)
        d_real_decision = self.dense(d_real_decision)

        d_loss = F.mse_loss(d_real_decision, self.real_label())
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_batch(self, batch):
        g_loss, d_loss_fake = self.train_generator(batch)
        d_loss_real = self.train_discriminator_real(batch)

        d_loss = (d_loss_real + d_loss_fake) / 2
        return [g_loss, d_loss]

    def loss_names_and_groups(_):
        return ['g_loss', 'd_loss'], {'GAN': ['g_loss', 'd_loss']}

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        x = self.dense(x)
        return x
