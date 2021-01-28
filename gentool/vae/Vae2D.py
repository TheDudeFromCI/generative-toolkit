import numpy as np
from math import log

import torch
from torchinfo import summary
from torch.cuda import FloatTensor
from torch.autograd.variable import Variable
from pywick.optimizers.rangerlars import RangerLars

from gentool.vae.Decoder import Decoder
from gentool.vae.Encoder import Encoder
from gentool.ModelBase import ImageModelBase
from gentool.vae.HyperParameters import Vae2DHyperParameters

from gentool.tools.PreGenDataset import load_dataset


class Vae2D(ImageModelBase):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        super().__init__(load_dataset(hyper_parameters.image_folder, hyper_parameters.batch_size))

        self.hyper_parameters = hyper_parameters
        self.encoder = Encoder(hyper_parameters)
        self.decoder = Decoder(hyper_parameters)

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        lr = hyper_parameters.learning_rate
        self.optimizer = RangerLars(self.parameters(), lr=lr, betas=betas)

        self.cuda()

        c, s = hyper_parameters.image_channels, hyper_parameters.image_size
        summary(self, (1, c, s, s), depth=4)

        self.avg_loss = []

    def forward(self, x):
        z, _, _ = self.encoder(x)
        return self.decoder(z)

    def sample_images(self, count):
        batch = next(self.dataloader)

        generated = self.forward(batch)
        images = [val for pair in zip(batch, generated) for val in pair]

        z = Variable(FloatTensor(np.random.normal(0, 1, (18, self.hyper_parameters.latent_dim))))
        images = images + [r for r in self.decoder(z)]

        return images, 12

    def train_batch(self):
        recon_loss_total = 0
        kld_loss_total = 0

        samples = 1
        for _ in range(samples):
            batch = next(self.dataloader)

            z, mu, var = self.encoder(batch)
            generated = self.decoder(z)

            alpha = 10
            beta = 1

            t = generated - batch
            recon_loss = alpha * t + torch.log(1 + torch.exp(-2 * alpha * t)) - log(2)
            recon_loss = (1 / alpha) * recon_loss.mean()

            kld_loss = -0.5 * torch.mean(1 + var - mu ** 2 - var.exp()) * self.hyper_parameters.kld_weight * beta
            loss = recon_loss + kld_loss
            loss.backward()

            recon_loss_total += recon_loss.item() / samples
            kld_loss_total += kld_loss.item() / samples

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.avg_loss.append(recon_loss_total + kld_loss_total)
        if len(self.avg_loss) > 10:
            self.avg_loss.pop(0)
        avg_loss = sum(self.avg_loss) / len(self.avg_loss)

        return '[recon_loss: {:.4f}, kld_loss: {:.4f}, avg_loss: {:.6f}]'.format(recon_loss_total, kld_loss_total, avg_loss)
