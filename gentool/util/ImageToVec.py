from gentool.util.ResidualBlock import ResidualBlock
from torch import nn


class ImageToVec(nn.Module):
    def __init__(self, image_size, image_channels, layers_per_size, initial_channels=4,
                 activation=nn.LeakyReLU(inplace=True), dense_layers=(64), output_activation=nn.Tanh()):
        super().__init__()

        self.image_size = image_size
        self.layers_per_size = layers_per_size
        self.initial_channels = initial_channels
        self.dense_layers = dense_layers

        blocks = []
        blocks.append(ResidualBlock(image_channels, initial_channels, image_size,
                                    activation=activation, normalization='group'))

        channels = initial_channels
        while image_size > 1:
            image_size >>= 1

            for i in range(layers_per_size):
                blocks.append(ResidualBlock(channels, channels, image_size,
                                            activation=activation, normalization='group'))

                if i == layers_per_size - 1:
                    c_old, channels = channels, int(channels * 2)
                    blocks.append(nn.Conv2d(c_old, channels, 3, 2, 1))

        blocks.append(nn.Flatten())

        for index, dense in enumerate(dense_layers):
            last_layer = dense_layers[index - 1] if index > 0 else channels
            blocks.append(nn.Linear(last_layer, dense))

            if index == len(dense_layers) - 1:
                blocks.append(output_activation)
            else:
                blocks.append(activation)

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)
