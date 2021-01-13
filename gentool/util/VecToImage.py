from math import log2

from gentool.util.ResidualBlock import ResidualBlock, get_normalization
from torch import nn


class VecToImage(nn.Module):
    def __init__(self, image_size, image_channels, layers_per_size, initial_channels=4,
                 activation=nn.LeakyReLU(inplace=True), output_activation=nn.Tanh(),
                 dropout=0.4, kernel=3, normalization='group', min_size=4):
        super().__init__()

        self.image_size = image_size
        self.layers_per_size = layers_per_size
        self.initial_channels = initial_channels
        self.kernel = kernel

        blocks = []

        max_channels = initial_channels << int(log2(image_size) - log2(min_size) + 1)
        blocks.append(nn.Unflatten(dim=1, unflattened_size=(max_channels, min_size, min_size)))

        channels = max_channels
        while min_size < image_size:
            min_size <<= 1

            for i in range(layers_per_size):
                if i == 0:
                    c_old, channels = channels, int(channels / 2)
                    blocks.append(nn.ConvTranspose2d(c_old, channels, 4, 2, 1))
                    blocks.append(get_normalization(normalization, channels, min_size, True))
                    blocks.append(activation)
                    blocks.append(nn.Dropout2d(dropout))

                blocks.append(ResidualBlock(channels, channels, min_size, skip_connections=False,
                                            activation=activation, normalization=normalization, dropout=dropout,
                                            kernel=kernel))

        blocks.append(nn.Conv2d(channels, image_channels, kernel, 1, int(kernel/2)))
        blocks.append(get_normalization(normalization, image_channels, image_size, True))
        blocks.append(output_activation)

        self.conv = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
