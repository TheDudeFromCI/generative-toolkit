from torch import nn

from gentool.util.ImageGenerator import ImageGenerator
from gentool.vae.HyperParameters import Vae2DHyperParameters


class Decoder(nn.Module):
    def __init__(self, hyper_parameters: Vae2DHyperParameters):
        super().__init__()

        self.module = ImageGenerator(hyper_parameters.image_size,
                                     hyper_parameters.image_channels,
                                     initial_channels=hyper_parameters.decoder_initial_channels,
                                     kernel=hyper_parameters.kernel,
                                     dropout=hyper_parameters.dropout,
                                     normalization=hyper_parameters.normalization,
                                     min_size=4,
                                     activation=nn.LeakyReLU(hyper_parameters.leaky_relu_slope, inplace=True),
                                     output_activation=nn.Tanh(),
                                     normalize_last=False,
                                     bias=hyper_parameters.bias_neurons)

    def forward(self, x):
        return self.module(x)
