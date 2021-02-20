from math import log2

from torch import nn
from torch.optim.adam import Adam

from Blocks import ResidualBlock, get_activation
from SubModuleBase import SubModuleBase


class ImgToImgDecoder(SubModuleBase):
    def create_network(self, config):
        self.image_channels = config['image_channels']
        self.image_size = config['image_size']

        learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
        beta1 = config['beta1'] if 'beta1' in config else 0.9
        beta2 = config['beta2'] if 'beta2' in config else 0.999

        model_channels = config['model_channels']
        skip_blocks = config['skip_blocks'] if 'skip_blocks' in config else 1
        kernel = config['kernel'] if 'kernel' in config else 3

        normalization = config['normalization']
        activation = config['activation']
        output_activation = config['output_activation'] if 'output_activation' in config else activation

        assert skip_blocks >= 1, "Cannot have negative skip layers!"
        assert kernel >= 1, "Cannot have negative kernel size!"
        assert kernel % 2 == 1, "Kernel must be an odd number value!"
        assert self.image_size >= 4, "Image size must be at least 4!"

        def res_block():
            blocks = []
            for index in range(skip_blocks):
                in_channels = self.image_channels if index == 0 else model_channels
                out_channels = self.image_channels if index == skip_blocks - 1 else model_channels
                blocks.append(ResidualBlock(in_channels, out_channels,
                                            self.image_size, kernel, normalization, activation))

            return blocks

        self.model = nn.Sequential(
            *res_block(),
            get_activation(output_activation),
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2))

    def forward(self, x):
        return self.model(x)
