import torch
from torch import nn
from pywick.functions.mish import Mish

from gentool.util.ImageToVec import ImageToVec
from gentool.vae.HyperParameters import Vae2DHyperParameters


class Encoder(nn.Module):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        super().__init__()

        self.module = ImageToVec(hyper_parameters.image_size,
                                 hyper_parameters.image_channels,
                                 hyper_parameters.encoder_layers_per_size,
                                 initial_channels=hyper_parameters.encoder_initial_channels,
                                 kernel=hyper_parameters.kernel,
                                 dropout=hyper_parameters.dropout,
                                 normalization=hyper_parameters.normalization,
                                 min_size=4,
                                 activation=Mish(),
                                 output_activation=Mish(),
                                 normalize_last=True,
                                 bias=True)

        self.mu = nn.Linear(hyper_parameters.encoder_out, hyper_parameters.latent_dim)
        self.var = nn.Linear(hyper_parameters.encoder_out, hyper_parameters.latent_dim)

    def reparameterize(_, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.module(x)
        mu = self.mu(x)
        var = self.var(x)
        z = self.reparameterize(mu, var)

        return [z, mu, var]
