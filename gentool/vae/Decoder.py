from torch import nn
from pywick.functions.mish import Mish

from gentool.util.ImageGenerator import ImageGenerator
from gentool.vae.HyperParameters import Vae2DHyperParameters


class Decoder(nn.Module):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        super().__init__()

        self.module = ImageGenerator(hyper_parameters.image_size,
                                     hyper_parameters.image_channels,
                                     initial_channels=hyper_parameters.decoder_initial_channels,
                                     kernel=hyper_parameters.kernel,
                                     normalization=hyper_parameters.normalization,
                                     min_size=4,
                                     activation=Mish(),
                                     output_activation=nn.Tanh())

    def forward(self, x):
        return self.module(x)
