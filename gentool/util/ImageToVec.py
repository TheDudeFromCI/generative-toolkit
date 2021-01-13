from gentool.util.ResidualBlock import ResidualBlock, get_normalization
from torch import nn


class ImageToVec(nn.Module):
    def __init__(self, image_size, image_channels, layers_per_size, initial_channels=4,
                 activation=nn.LeakyReLU(inplace=True), output_activation=nn.Tanh(),
                 dropout=0.4, kernel=3, normalization='group', min_size=4, normalize_last=False):
        super().__init__()

        blocks = []
        blocks.append(nn.Conv2d(image_channels, initial_channels, kernel, 1, int(kernel/2)))
        blocks.append(get_normalization(normalization, initial_channels, image_size, True))
        blocks.append(activation)
        blocks.append(nn.Dropout2d(dropout))

        channels = initial_channels
        while image_size > min_size:
            image_size >>= 1

            for i in range(layers_per_size):
                blocks.append(ResidualBlock(channels, channels, image_size, kernel=kernel,
                                            activation=activation, normalization=normalization, dropout=dropout))

                if i == layers_per_size - 1:
                    c_old, channels = channels, int(channels * 2)
                    blocks.append(nn.Conv2d(c_old, channels, 3, 2, 1))

                    if image_size > min_size:
                        blocks.append(get_normalization(normalization, channels, image_size, True))
                        blocks.append(activation)
                        blocks.append(nn.Dropout2d(dropout))
                    else:
                        if normalize_last:
                            blocks.append(get_normalization(normalization, channels, image_size, True))
                        blocks.append(output_activation)

        for i in range(layers_per_size):
            blocks.append(ResidualBlock(channels, channels, image_size, kernel=kernel,
                                        activation=activation, normalization=normalization, dropout=dropout))

        blocks.append(nn.Flatten())
        self.conv = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x
