from torch import nn

from gentool.util.ImageGenerator import ImageGenerator
from gentool.gan.HyperParameters import Gan2DHyperParameters


class Generator(nn.Module):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        super().__init__()

        self.module = ImageGenerator(hyper_parameters.image_size,
                                     hyper_parameters.image_channels,
                                     initial_channels=hyper_parameters.gen_initial_channels,
                                     kernel=hyper_parameters.kernel,
                                     dropout=hyper_parameters.dropout,
                                     normalization=hyper_parameters.normalization,
                                     min_size=4,
                                     activation=nn.ReLU(inplace=True),
                                     output_activation=nn.Tanh(),
                                     normalize_last=False,
                                     bias=hyper_parameters.bias_neurons)

    def forward(self, x):
        return self.module(x)
