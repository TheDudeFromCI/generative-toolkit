from math import log2

from torch import nn
from pywick.functions.mish import Mish

from gentool.util.ResidualBlock import get_normalization


class ImageGenerator(nn.Module):
    def __init__(self, image_size, image_channels, initial_channels=4,
                 activation=Mish(), output_activation=nn.Tanh(),
                 kernel=3, normalization='group', min_size=4, bias=False):
        super().__init__()

        blocks = []

        max_channels = initial_channels << int(log2(image_size) - log2(min_size))
        blocks.append(nn.Unflatten(dim=1, unflattened_size=(max_channels, min_size, min_size)))

        channels = max_channels
        while min_size < image_size:
            min_size <<= 1

            c_old, channels = channels, int(channels / 2)
            blocks.append(nn.ConvTranspose2d(c_old, channels, 4, 2, 1, bias=bias))
            blocks.append(get_normalization(normalization, channels, min_size, False))
            blocks.append(activation)

            blocks.append(nn.Conv2d(channels, channels, kernel, 1, int(kernel/2), bias=bias))
            blocks.append(get_normalization(normalization, channels, min_size, False))
            blocks.append(activation)

            blocks.append(nn.Conv2d(channels, channels, kernel, 1, int(kernel/2), bias=bias))
            blocks.append(get_normalization(normalization, channels, min_size, False))
            blocks.append(activation)

        blocks.append(nn.Conv2d(channels, image_channels, kernel, 1, int(kernel/2), bias=bias))
        blocks.append(output_activation)

        self.conv = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
