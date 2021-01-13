from math import log2

from gentool.util.ResidualBlock import ResidualBlock
from torch import nn


class VecToImage(nn.Module):
    def __init__(self, image_size, image_channels, layers_per_size, initial_channels=4,
                 activation=nn.LeakyReLU(inplace=True), dense_layers=(64), output_activation=nn.Tanh(),
                 dropout=0.4, kernel=3):
        super().__init__()

        self.image_size = image_size
        self.layers_per_size = layers_per_size
        self.initial_channels = initial_channels
        self.dense_layers = dense_layers
        self.kernel = kernel

        blocks = []

        max_channels = initial_channels << int(log2(image_size))

        for index, dense in enumerate(dense_layers):
            next_layer = dense_layers[index + 1] if index < len(dense_layers) - 1 else max_channels
            blocks.append(nn.Linear(dense, next_layer))
            blocks.append(activation)
            blocks.append(nn.Dropout(dropout))

        blocks.append(nn.Unflatten(dim=1, unflattened_size=(max_channels, 1, 1)))

        channels = max_channels
        min_size = 1
        while min_size < image_size:
            min_size <<= 1

            for i in range(layers_per_size):
                if i == 0:
                    c_old, channels = channels, int(channels / 2)
                    blocks.append(nn.ConvTranspose2d(c_old, channels, 4, 2, 1))

                blocks.append(ResidualBlock(channels, channels, min_size, skip_connections=False,
                                            activation=activation, normalization='group', dropout=dropout,
                                            kernel=kernel))

        blocks.append(nn.Conv2d(channels, image_channels, kernel, 1, int(kernel/2)))
        blocks.append(output_activation)

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)
