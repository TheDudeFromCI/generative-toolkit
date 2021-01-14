from gentool.util.ImageGenerator import ImageGenerator
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
        self.dense = self._build_dense()

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        self.optimizer_g = Adam(self.generator.parameters(), lr=hyper_parameters.learning_rate, betas=betas)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=hyper_parameters.learning_rate, betas=betas)

        self.cuda()
        summary(self, (1, hyper_parameters.latent_dim), depth=3)

    def _build_generator(self):
        return ImageGenerator(self.hyper_parameters.image_size,
                              self.hyper_parameters.image_channels,
                              initial_channels=self.hyper_parameters.gen_initial_channels,
                              kernel=self.hyper_parameters.kernel,
                              dropout=self.hyper_parameters.dropout,
                              normalization=self.hyper_parameters.normalization,
                              min_size=4,
                              activation=nn.ReLU(inplace=True),
                              output_activation=nn.Tanh(),
                              normalize_last=False,
                              bias=self.hyper_parameters.bias_neurons)

    def _build_discriminator(self):
        return ImageToVec(self.hyper_parameters.image_size,
                          self.hyper_parameters.image_channels,
                          self.hyper_parameters.discriminator_layers_per_size,
                          initial_channels=self.hyper_parameters.dis_initial_channels,
                          kernel=self.hyper_parameters.kernel,
                          dropout=self.hyper_parameters.dropout,
                          normalization=self.hyper_parameters.normalization,
                          min_size=1,
                          activation=nn.LeakyReLU(self.hyper_parameters.leaky_relu_slope, inplace=True),
                          output_activation=nn.LeakyReLU(self.hyper_parameters.leaky_relu_slope, inplace=True),
                          normalize_last=True,
                          bias=self.hyper_parameters.bias_neurons)

    def _build_dense(self):
        return nn.Sequential(
            nn.Linear(self.hyper_parameters.discriminator_out, 1, bias=self.hyper_parameters.bias_neurons),
            nn.LeakyReLU(self.hyper_parameters.leaky_relu_slope, inplace=True)
        )

    def sample_images(self, count):
        z = Variable(FloatTensor(np.random.normal(0, 1, (count, self.hyper_parameters.latent_dim))))
        generated = self.generator(z)
        batch = next(self.dataloader)

        return torch.cat([generated, batch])

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

        fake_input_noise = torch.randn_like(batch) * 0.1
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)
        dg_fake_decision = self.dense(dg_fake_decision)

        g_loss = F.mse_loss(dg_fake_decision, self.real_label())
        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    def train_discriminator_fake(self, batch):
        self.optimizer_d.zero_grad()

        fake_input_noise = torch.randn_like(batch) * 0.1
        g_fake_data = self.generator(self.z_noise()) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)
        dg_fake_decision = self.dense(dg_fake_decision)

        d_loss = F.mse_loss(dg_fake_decision, self.fake_label())
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_discriminator_real(self, batch):
        self.optimizer_d.zero_grad()

        real_input_noise = torch.randn_like(batch) * 0.1
        d_real_decision = self.discriminator(batch + real_input_noise)
        d_real_decision = self.dense(d_real_decision)

        d_loss = F.mse_loss(d_real_decision, self.real_label())
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_batch(self, batch):
        d_loss_real = self.train_discriminator_real(batch)
        d_loss_fake = self.train_discriminator_fake(batch)

        g_loss = self.train_generator(batch)
        d_loss = (d_loss_real + d_loss_fake) / 2

        return [g_loss, d_loss]

    def loss_names_and_groups(_):
        return ['g_loss', 'd_loss'], {'GAN': ['g_loss', 'd_loss']}

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        x = self.dense(x)
        return x
