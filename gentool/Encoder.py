from math import log2

from torch import nn
from torch.optim.adam import Adam

from Blocks import ResidualBlockDown, get_normalization_1d, get_activation
from SubModuleBase import SubModuleBase


class Encoder(SubModuleBase):
    def create_network(self, config):
        self.image_channels = config['image_channels']
        self.image_size = config['image_size']
        self.latent_dim = config['latent_dim']

        learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
        beta1 = config['beta1'] if 'beta1' in config else 0.9
        beta2 = config['beta2'] if 'beta2' in config else 0.999

        downsample_method = config['downsample_method']
        model_channels = config['model_channels']
        max_channels = config['max_channels']
        skip_blocks = config['skip_blocks'] if 'skip_blocks' in config else 1
        kernel = config['kernel'] if 'kernel' in config else 3
        dense_layers = config['dense_layers'] if 'dense_layers' in config else 1
        dense_width = config['dense_width'] if 'dense_width' in config else self.latent_dim
        skip_blocks_doubling = config['skip_blocks_doubling'] if 'skip_blocks_doubling' in config else False

        normalization = config['normalization']
        activation = config['activation']
        output_activation = config['output_activation'] if 'output_activation' in config else activation

        assert dense_layers >= 1, "Cannot have negative dense layers!"
        assert skip_blocks >= 1, "Cannot have negative skip layers!"
        assert kernel >= 1, "Cannot have negative kernel size!"
        assert kernel % 2 == 1, "Kernel must be an odd number value!"
        assert self.image_size >= 4, "Image size must be at least 4!"
        assert self.latent_dim >= 1, "Latent dimensions must be at least 1!"

        downsamples = int(log2(self.image_size)) - 2

        def res_block():
            blocks = []
            for index in range(downsamples):
                in_channels = min((1 << index) * 1 * model_channels, max_channels)
                out_channels = min((1 << index) * 2 * model_channels, max_channels)

                skips = skip_blocks
                if skip_blocks_doubling:
                    skips <<= index

                blocks.append(ResidualBlockDown(in_channels, out_channels,
                                                self.image_size >> index, downsample_method, kernel, normalization, activation))

                for _ in range(skips - 1):
                    blocks.append(ResidualBlockDown(out_channels, out_channels,
                                                    self.image_size >> (index + 1), 'none', kernel, normalization, activation))

            return blocks

        def dense_block():
            blocks = []

            dense_channels = min((1 << downsamples) * model_channels, max_channels) * 4 * 4

            dense_out = dense_width if dense_layers > 1 else self.latent_dim
            blocks.append(nn.Linear(dense_channels, dense_out))

            if dense_layers > 1:
                blocks.append(get_normalization_1d(normalization, dense_out))
                blocks.append(get_activation(activation))

            for i in range(dense_layers - 1):
                dense_out = dense_width if i < dense_layers - 2 else self.latent_dim
                blocks.append(nn.Linear(dense_width, dense_out))

                if i < dense_layers - 2:
                    blocks.append(get_normalization_1d(normalization, dense_out))
                    blocks.append(get_activation(activation))

            blocks.append(get_activation(output_activation))
            return blocks

        self.model = nn.Sequential(
            nn.Conv2d(self.image_channels, model_channels, kernel, 1, kernel // 2),
            *res_block(),

            nn.Flatten(),
            *dense_block(),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def forward(self, x):
        return self.model(x)
