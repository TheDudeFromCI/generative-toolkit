from math import log2

from torch import nn
from torch.optim.adam import Adam

from Blocks import ResidualBlockUp, get_normalization, get_normalization_1d, get_activation
from SubModuleBase import SubModuleBase


class Decoder(SubModuleBase):
    def create_network(self, config):
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
        dense_width = config['dense_width'] if 'dense_width' in config else self.latent_dim

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
                image_size = 4 << index

                for _ in range(skip_blocks - 1):
                    blocks.append(ResidualBlockUp(in_channels, in_channels, image_size,
                                                  'none', kernel, normalization, activation))

                blocks.append(ResidualBlockUp(in_channels, out_channels, image_size,
                                              upsample_method, kernel, normalization, activation))
            return blocks

        def dense_block():
            blocks = []

            init_channels = min((1 << upsamples) * model_channels, max_channels)
            for i in range(dense_layers):
                dense_in = self.latent_dim if i == 0 else dense_width
                dense_out = dense_width if i < dense_layers - 1 else init_channels * 4 * 4
                blocks.append(nn.Linear(dense_in, dense_out))

                if i < dense_layers - 1:
                    blocks.append(get_normalization_1d(normalization, dense_out))
                    blocks.append(get_activation(activation))

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

    def forward(self, x):
        return self.model(x)
