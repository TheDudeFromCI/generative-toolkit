from math import log2

from torch import nn
from torch.optim.adam import Adam

from Blocks import ResidualBlockDown, get_normalization_1d, get_activation
from SubModuleBase import SubModuleBase


class DualEncoder(SubModuleBase):
    def create_network(self, config):
        self.image_channels = config['image_channels']
        self.image_size = config['image_size']

        self.latent_dim_1 = config['latent_dim_1']
        self.latent_dim_2 = config['latent_dim_2']

        learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
        beta1 = config['beta1'] if 'beta1' in config else 0.9
        beta2 = config['beta2'] if 'beta2' in config else 0.999

        downsample_method = config['downsample_method']
        model_channels = config['model_channels']
        max_channels = config['max_channels']
        skip_blocks = config['skip_blocks'] if 'skip_blocks' in config else 1
        kernel = config['kernel'] if 'kernel' in config else 3

        dense_layers_1 = config['dense_layers_1'] if 'dense_layers_1' in config else 1
        dense_layers_2 = config['dense_layers_2'] if 'dense_layers_2' in config else 1

        normalization = config['normalization']
        activation = config['activation']

        output_activation_1 = config['output_activation_1'] if 'output_activation_1' in config else activation
        output_activation_2 = config['output_activation_2'] if 'output_activation_2' in config else activation

        assert dense_layers_1 >= 1, "Cannot have negative dense layers!"
        assert dense_layers_2 >= 1, "Cannot have negative dense layers!"
        assert skip_blocks >= 1, "Cannot have negative skip layers!"
        assert kernel >= 1, "Cannot have negative kernel size!"
        assert kernel % 2 == 1, "Kernel must be an odd number value!"
        assert self.image_size >= 4, "Image size must be at least 4!"
        assert self.latent_dim_1 >= 1, "Latent dimensions must be at least 1!"
        assert self.latent_dim_2 >= 1, "Latent dimensions must be at least 1!"

        downsamples = int(log2(self.image_size)) - 2

        def res_block():
            blocks = []
            for index in range(downsamples):
                in_channels = min((1 << index) * 1 * model_channels, max_channels)
                out_channels = min((1 << index) * 2 * model_channels, max_channels)

                blocks.append(ResidualBlockDown(in_channels, out_channels,
                                                self.image_size >> index, downsample_method, kernel, normalization, activation))

                for _ in range(skip_blocks - 1):
                    blocks.append(ResidualBlockDown(out_channels, out_channels,
                                                    self.image_size >> (index + 1), 'none', kernel, normalization, activation))

            return blocks

        def dense_block(layers, latent, output_activation):
            blocks = []

            dense_channels = min((1 << downsamples) * model_channels, max_channels) * 4 * 4
            blocks.append(nn.Linear(dense_channels, latent))

            if layers > 1:
                blocks.append(get_normalization_1d(normalization, latent))
                blocks.append(get_activation(activation))

            for i in range(layers - 1):
                blocks.append(nn.Linear(latent, latent))

                if i < layers - 2:
                    blocks.append(get_normalization_1d(normalization, latent))
                    blocks.append(get_activation(activation))

            blocks.append(get_activation(output_activation))
            return blocks

        self.model = nn.Sequential(
            nn.Conv2d(self.image_channels, model_channels, kernel, 1, kernel // 2),
            *res_block(),

            nn.Flatten(),
        )

        self.latent_1_model = nn.Sequential(*dense_block(dense_layers_1, self.latent_dim_1, output_activation_1))
        self.latent_2_model = nn.Sequential(*dense_block(dense_layers_2, self.latent_dim_2, output_activation_2))

        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def forward(self, x):
        x = self.model(x)
        return self.latent_1_model(x), self.latent_2_model(x)
