import numpy as np
from math import log

import torch
from torch import nn
from torch.optim import Adam
from torchinfo import summary
from torch.cuda import FloatTensor
from torch.autograd.variable import Variable
from pywick.optimizers.rangerlars import RangerLars

from gentool.ModelBase import ImageModelBase
from gentool.vae.HyperParameters import Vae2DHyperParameters

from gentool.tools.PreGenDataset import load_dataset


class Vae2D(ImageModelBase):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        super().__init__(load_dataset(hyper_parameters.image_folder, hyper_parameters.batch_size))

        self.hyper_parameters = hyper_parameters
        self.encoder = hyper_parameters.encoder
        self.decoder = hyper_parameters.decoder

        self.mu = hyper_parameters.mu
        self.var = hyper_parameters.var

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        lr = hyper_parameters.learning_rate
        self.optimizer = Adam(self.parameters(), lr=lr, betas=betas)

        self.cuda()

        c, s = hyper_parameters.image_channels, hyper_parameters.image_size
        summary(self, (1, c, s, s), depth=4)

    def reparameterize(_, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.encoder(x)
        x = self.reparameterize(self.mu(x), self.var(x))
        return self.decoder(x)

    def sample_images(self):
        batch = next(self.dataloader)

        generated = self.forward(batch)
        images = [val for pair in zip(batch, generated) for val in pair]

        z = Variable(FloatTensor(np.random.normal(0, 1, (18, self.hyper_parameters.latent_dim))))
        images = images + [r for r in self.decoder(z)]

        return images, 12

    def train_batch(self):
        recon_loss_total = 0
        kld_loss_total = 0

        for _ in range(self.hyper_parameters.gradient_updates):
            batch = next(self.dataloader)

            x = self.encoder(batch)
            mu = self.mu(x)
            var = self.var(x)
            z = self.reparameterize(mu, var)
            generated = self.decoder(z)

            alpha = 10
            beta = 1

            t = generated - batch
            recon_loss = alpha * t + torch.log(1 + torch.exp(-2 * alpha * t)) - log(2)
            recon_loss = (1 / alpha) * recon_loss.mean()

            kld_loss = -0.5 * torch.mean(1 + var - mu ** 2 - var.exp()) * self.hyper_parameters.kld_weight * beta
            loss = recon_loss + kld_loss
            loss.backward()

            recon_loss_total += recon_loss.item() / self.hyper_parameters.gradient_updates
            kld_loss_total += kld_loss.item() / self.hyper_parameters.gradient_updates

        self.optimizer.step()
        self.optimizer.zero_grad()

        return '[recon_loss: {:.4f}, kld_loss: {:.4f}]'.format(recon_loss_total, kld_loss_total)
