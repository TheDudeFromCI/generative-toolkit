from math import log2

from torch import nn
from torch.optim.adam import Adam

from Blocks import ResidualBlockUp, get_normalization, get_normalization_1d, get_activation
from SubModuleBase import SubModuleBase


class Decoder(SubModuleBase):
    def __init__(self, config):
        super().__init__(config['save_file'])

        self.image_channels = config['image_channels']
        self.image_size = config['image_size']
        self.latent_dim = config['latent_dim']

        learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
        beta1 = config['beta1'] if 'beta1' in config else 0.9
        beta2 = config['beta2'] if 'beta2' in config else 0.999

        upsample_method = config['upsample_method']
        model_channels = config['model_channels']
        max_channels = config['max_channels']
        skip_blocks = config['skip_blocks'] if 'skip_blocks' in config else 1
        kernel = config['kernel'] if 'kernel' in config else 3
        dense_layers = config['dense_layers'] if 'dense_layers' in config else 1

        normalization = config['normalization']
        activation = config['activation']
        output_activation = config['output_activation'] if 'output_activation' in config else activation

        assert dense_layers >= 1, "Cannot have negative dense layers!"
        assert skip_blocks >= 1, "Cannot have negative skip layers!"
        assert kernel >= 1, "Cannot have negative kernel size!"
        assert kernel % 2 == 1, "Kernel must be an odd number value!"
        assert self.image_size >= 4, "Image size must be at least 4!"
        assert self.latent_dim >= 1, "Latent dimensions must be at least 1!"

        upsamples = int(log2(self.image_size)) - 2

        def res_block():
            blocks = []
            for index in range(upsamples - 1, -1, -1):
                in_channels = min((1 << index) * 2 * model_channels, max_channels)
                out_channels = min((1 << index) * 1 * model_channels, max_channels)

                for _ in range(skip_blocks - 1):
                    blocks.append(ResidualBlockUp(in_channels, in_channels, 'none', kernel))

                blocks.append(ResidualBlockUp(in_channels, out_channels, upsample_method, kernel))
            return blocks

        def dense_block():
            blocks = []

            for _ in range(dense_layers - 1):
                blocks.append(nn.Linear(self.latent_dim, self.latent_dim))
                blocks.append(get_normalization_1d(normalization, self.latent_dim))
                blocks.append(get_activation(activation))

            init_channels = min((1 << upsamples) * model_channels, max_channels)
            blocks.append(nn.Linear(self.latent_dim, init_channels * 4 * 4))
            blocks.append(nn.Unflatten(1, (init_channels, 4, 4)))
            return blocks

        self.model = nn.Sequential(
            *dense_block(),
            *res_block(),

            get_normalization(normalization, model_channels, self.image_size),
            get_activation(activation),

            nn.Conv2d(model_channels, self.image_channels, kernel, 1, kernel // 2),
            get_activation(output_activation),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.try_load()

    def forward(self, x):
        return self.model(x)
