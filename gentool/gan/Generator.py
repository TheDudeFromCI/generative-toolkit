import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.nn import functional as F

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
                                     activation=nn.LeakyReLU(hyper_parameters.leaky_relu_slope, inplace=True),
                                     output_activation=nn.Tanh(),
                                     normalize_last=False,
                                     bias=hyper_parameters.bias_neurons)

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        self.optimizer = Adam(self.parameters(), lr=hyper_parameters.gen_learning_rate, betas=betas)

    def forward(self, x):
        return self.module(x)
