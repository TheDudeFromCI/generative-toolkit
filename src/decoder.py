from torch import nn

from src.residual_block import ResidualBlock


class Upscaler(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        super().__init__()

        blocks = []
        blocks.append(nn.Upsample(scale_factor=2))

        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for i in range(layers):
            out = out_channels if i == layers - 1 else in_channels
            blocks.append(ResidualBlock(in_channels, out, activation))

        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, image_size, channels, latent_dim, activation, layers_per_size=2, channel_scaling=2):
        super().__init__()

        assert (image_size & (image_size - 1) == 0) and image_size > 0, \
            'image_size must be a power of 2!'

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        def scale(x): return int(x * channel_scaling)

        blocks = [activation]
        while image_size > 1:
            image_size /= 2
            blocks.insert(0, Upscaler(
                scale(channels), channels, layers=layers_per_size))
            channels = scale(channels)

        blocks.insert(0, nn.Unflatten(
            dim=1, unflattened_size=(channels, 1, 1)))

        self.layers = nn.Sequential(*blocks)
        self.max_channels = channels

        self.dense = nn.Sequential(
            nn.Linear(latent_dim, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        x = self.dense(x)
        return self.layers(x)
