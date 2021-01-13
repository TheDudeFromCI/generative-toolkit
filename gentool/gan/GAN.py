from gentool.ModelBase import ImageModelBase

import numpy as np
from math import floor

import torch
from torch import nn
from torch.cuda import FloatTensor
from torch.autograd import Variable
from torch.optim.adam import Adam
import torch.nn.functional as F
import torchvision.transforms as T
from torchinfo import summary

from gentool.util.ImageToVec import ImageToVec
from gentool.util.VecToImage import VecToImage
from gentool.database.ImageDataLoader import image_dataloader


class GanHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 3
        self.generator_layers_per_size = 3
        self.discriminator_layers_per_size = 3
        self.latent_dim = 32
        self.initial_channels = 4
        self.learning_rate = 1e-4
        self.kernel = 5
        self.batch_size = 25
        self.image_folders = []
        self.data_augmentation = T.ToTensor()
        self.data_workers = 4
        self.generator_dense_layers = (32, 32)
        self.discriminator_dense_layers = (32, 16, 1)
        self.dropout = 0.4


class GAN(ImageModelBase):
    def __init__(self, hyper_parameters: GanHyperParameters):
        dataloader = image_dataloader(hyper_parameters.image_folders, hyper_parameters.batch_size,
                                      hyper_parameters.data_augmentation, workers=hyper_parameters.data_workers)
        super().__init__(dataloader)

        self.hyper_parameters = hyper_parameters

        self.generator = VecToImage(hyper_parameters.image_size,
                                    hyper_parameters.image_channels,
                                    hyper_parameters.generator_layers_per_size,
                                    initial_channels=hyper_parameters.initial_channels,
                                    dense_layers=hyper_parameters.generator_dense_layers,
                                    kernel=hyper_parameters.kernel,
                                    dropout=hyper_parameters.dropout)

        self.discriminator = ImageToVec(hyper_parameters.image_size,
                                        hyper_parameters.image_channels,
                                        hyper_parameters.discriminator_layers_per_size,
                                        initial_channels=hyper_parameters.initial_channels,
                                        dense_layers=hyper_parameters.discriminator_dense_layers,
                                        output_activation=nn.Sigmoid(),
                                        kernel=hyper_parameters.kernel,
                                        dropout=hyper_parameters.dropout)

        self.optimizer_g = Adam(self.generator.parameters(), lr=hyper_parameters.learning_rate)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=hyper_parameters.learning_rate)

        self.cuda()
        summary(self, depth=1)

    def sample_images(self, count):
        z = Variable(FloatTensor(np.random.normal(0, 1, (count, self.latent_dim))))
        return self.generator(z)

    def train_batch(self, batch):
        # Initialize
        batch_size = len(batch)
        zeros = Variable(FloatTensor(np.zeros((batch_size, 1))))
        ones = Variable(FloatTensor(np.ones((batch_size, 1))))
        z_noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
        real_input_noise = torch.randn_like(batch)
        fake_input_noise = torch.randn_like(batch)

        # Update generator
        self.optimizer_g.zero_grad()
        g_fake_data = self.generator(z_noise) + fake_input_noise
        dg_fake_decision = self.discriminator(g_fake_data)

        g_loss = F.mse_loss(dg_fake_decision, ones)
        g_loss.backward()
        self.optimizer_g.step()

        # Update discriminator
        g_fake_data = Variable(g_fake_data.data)
        total_input = torch.cat([batch + real_input_noise, g_fake_data])
        total_target = torch.cat([ones, zeros])

        self.optimizer_d.zero_grad()
        d_real_decision = self.discriminator(total_input)

        d_loss = F.mse_loss(d_real_decision, total_target)
        d_loss.backward()
        self.optimizer_d.step()

        return [g_loss.item(), d_loss.item()]

    def loss_names_and_groups(self):
        return ['g_loss', 'd_loss'], {'GAN': ['g_loss', 'd_loss']}
