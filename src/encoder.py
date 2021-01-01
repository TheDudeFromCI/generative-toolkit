import torch
from torch import nn

from src.residual_block import ResidualBlock


class Downscaler(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        super().__init__()

        blocks = []

        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(layers):
            out = out_channels if i == layers - 1 else in_channels
            blocks.append(ResidualBlock(in_channels, out, activation))

        blocks.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, image_size, channels, layers_per_size=2, channel_scaling=2):
        super().__init__()

        assert (image_size & (image_size - 1) == 0) and image_size > 0, \
            'image_size must be a power of 2!'

        self.image_size = image_size
        self.channels = channels

        def scale(x): return int(x * channel_scaling)

        blocks = []
        while image_size > 1:
            image_size /= 2
            blocks.append(Downscaler(
                channels, scale(channels), layers=layers_per_size))
            channels = scale(channels)

        blocks.append(nn.Flatten())

        self.layers = nn.Sequential(*blocks)
        self.max_channels = channels

    def forward(self, x):
        return self.layers(x)
