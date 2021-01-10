from gentool.util.ResidualBlock import ResidualBlock
from torch import nn


class ImageToVec(nn.Module):
    def __init__(self, image_size, layers_per_size, channel_scaling=2, initial_channels=4, activation=nn.LeakyReLU(), dense_layers=(64)):
        super().__init__()

        self.image_size = image_size
        self.layers_per_size = layers_per_size
        self.channel_scaling = channel_scaling
        self.initial_channels = initial_channels
        self.dense_layers = dense_layers

        blocks = []
        channels = initial_channels
        while image_size > 1:
            image_size >>= 1

            for i in range(layers_per_size):
                out_channels = int(channels * channel_scaling) if i == layers_per_size - 1 else channels
                blocks.append(ResidualBlock(channels, out_channels, image_size,
                                            activation=activation, normalization='group'))

                channels = out_channels

        blocks.append(nn.Flatten())

        for index, dense in enumerate(dense_layers):
            next_layer = dense_layers[index + 1] if index > 0 else channels
            blocks.append(nn.Linear(dense, next_layer))

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)
