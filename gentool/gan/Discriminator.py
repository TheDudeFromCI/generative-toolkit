from torch import nn

from gentool.util.ImageToVec import ImageToVec
from gentool.gan.HyperParameters import Gan2DHyperParameters


class Discriminator(nn.Module):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        super().__init__()

        self.module = ImageToVec(hyper_parameters.image_size,
                                 hyper_parameters.image_channels,
                                 hyper_parameters.discriminator_layers_per_size,
                                 initial_channels=hyper_parameters.dis_initial_channels,
                                 kernel=hyper_parameters.kernel,
                                 dropout=hyper_parameters.dropout,
                                 normalization=hyper_parameters.normalization,
                                 min_size=1,
                                 activation=nn.LeakyReLU(hyper_parameters.leaky_relu_slope, inplace=True),
                                 output_activation=nn.LeakyReLU(hyper_parameters.leaky_relu_slope, inplace=True),
                                 normalize_last=True,
                                 bias=hyper_parameters.bias_neurons)

        self.dense = nn.Sequential(
            nn.Linear(hyper_parameters.discriminator_out, 64, bias=hyper_parameters.bias_neurons),
            nn.LeakyReLU(hyper_parameters.leaky_relu_slope, inplace=True),
            nn.Linear(64, 1, bias=hyper_parameters.bias_neurons),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.module(x)
        x = self.dense(x)
        return x
