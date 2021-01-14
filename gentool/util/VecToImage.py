from math import log2

from gentool.util.ResidualBlock import ResidualBlock, get_normalization
from torch import nn


class VecToImage(nn.Module):
    def __init__(self, image_size, image_channels, layers_per_size, initial_channels=4,
                 activation=nn.LeakyReLU(inplace=True), output_activation=nn.Tanh(),
                 dropout=0.4, kernel=3, normalization='group', min_size=4, normalize_last=False,
                 bias=False):
        super().__init__()

        blocks = []

        max_channels = initial_channels << int(log2(image_size) - log2(min_size))
        blocks.append(nn.Unflatten(dim=1, unflattened_size=(max_channels, min_size, min_size)))

        channels = max_channels
        while min_size < image_size:
            min_size <<= 1

            for i in range(layers_per_size):
                blocks.append(ResidualBlock(channels, channels, min_size, skip_connections=False,
                                            activation=activation, normalization=normalization, dropout=dropout,
                                            kernel=kernel, bias=bias))

                if i == layers_per_size - 1:
                    c_old, channels = channels, int(channels / 2)
                    blocks.append(nn.ConvTranspose2d(c_old, channels, 4, 2, 1, bias=bias))
                    blocks.append(get_normalization(normalization, channels, min_size, True))
                    blocks.append(nn.Dropout2d(dropout))
                    blocks.append(activation)

        for i in range(layers_per_size):
            blocks.append(ResidualBlock(channels, channels, min_size, skip_connections=False,
                                        activation=activation, normalization=normalization, dropout=dropout,
                                        kernel=kernel, bias=bias))

        blocks.append(nn.Conv2d(channels, image_channels, kernel, 1, int(kernel/2), bias=bias))

        if normalize_last:
            blocks.append(get_normalization(normalization, image_channels, image_size, True))
            blocks.append(nn.Dropout2d(dropout))

        blocks.append(output_activation)

        self.conv = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
