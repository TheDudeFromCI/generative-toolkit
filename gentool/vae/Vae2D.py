import numpy as np
from math import ceil, sqrt

import torch
from torchinfo import summary
from torch.optim.adam import Adam
from torch.cuda import FloatTensor
from torch.nn import functional as F
from torch.autograd.variable import Variable

from gentool.vae.Decoder import Decoder
from gentool.vae.Encoder import Encoder
from gentool.ModelBase import ImageModelBase
from gentool.database.ImageDataLoader import image_dataloader
from gentool.vae.HyperParameters import Vae2DHyperParameters


class Vae2D(ImageModelBase):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        dataloader = image_dataloader(hyper_parameters.image_folders, hyper_parameters.batch_size,
                                      hyper_parameters.data_augmentation, workers=hyper_parameters.data_workers)
        super().__init__(dataloader)
        self.hyper_parameters = hyper_parameters
        self.encoder = Encoder(hyper_parameters)
        self.decoder = Decoder(hyper_parameters)

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        self.optimizer = Adam(self.parameters(), lr=hyper_parameters.learning_rate, betas=betas)

        self.cuda()

        c, s = hyper_parameters.image_channels, hyper_parameters.image_size
        summary(self, (1, c, s, s), depth=4)

    def forward(self, x):
        z, _, _ = self.encoder(x)
        return self.decoder(z)

    def sample_images(self, count):
        z = Variable(FloatTensor(np.random.normal(0, 1, (count, self.hyper_parameters.latent_dim))))
        generated = self.decoder(z)

        samples = ceil(count / self.hyper_parameters.batch_size)
        extra = self.hyper_parameters.batch_size * samples - count * samples
        for i in range(samples):
            batch = next(self.dataloader)

            if i == samples - 1 and extra > 0:
                batch = batch[:extra]

            generated = torch.cat([generated, batch])

        return generated, int(sqrt(count) * 2)

    def train_batch(self):
        batch = next(self.dataloader)

        z, mu, var = self.encoder(batch)
        generated = self.decoder(z)

        recon_loss = F.mse_loss(generated, batch, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + var - mu.pow(2) - var.exp()) * self.hyper_parameters.kld_weight

        loss = recon_loss + kld_loss
        loss.backward()
        self.optimizer.step()

        return '[recon_loss: {:.4f}, kld_loss: {:.4f}]'.format(recon_loss.item(), kld_loss.item())
