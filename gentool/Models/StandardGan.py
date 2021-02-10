from math import log2

from torch import nn

from ModelBase import GanModelBase
from Database import numpy_dataloader


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, downsample_method, kernel):
        super().__init__()

        if downsample_method == 'none':
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            conv_downsample = nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2)

        elif downsample_method == 'avg_pool':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            )

            conv_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2),
                nn.AvgPool2d(2, 2),
            )

        elif downsample_method == 'max_pool':
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            )

            conv_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2),
                nn.MaxPool2d(2, 2),
            )

        elif downsample_method == 'stride':
            self.shortcut = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
            conv_downsample = nn.Conv2d(in_channels, out_channels, kernel, 2, kernel // 2)

        else:
            assert False, f"Unknown downsample method '{downsample_method}'!"

        self.conv = nn.Sequential(
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, in_channels, kernel, 1, kernel // 2),
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            conv_downsample,
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_method, kernel):
        super().__init__()

        if upsample_method == 'none':
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            conv_upsample = nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2)

        elif upsample_method == 'transpose':
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            conv_upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

        elif upsample_method == 'nearest':
            if in_channels == out_channels:
                self.shortcut = nn.UpsamplingNearest2d(scale_factor=2)
            else:
                self.shortcut = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                )

            conv_upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2),
                nn.UpsamplingNearest2d(scale_factor=2),
            )

        elif upsample_method == 'bilinear':
            if in_channels == out_channels:
                self.shortcut = nn.UpsamplingBilinear2d(scale_factor=2)
            else:
                self.shortcut = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                )

            conv_upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2),
                nn.UpsamplingBilinear2d(scale_factor=2),
            )

        else:
            assert False, f"Unknown upsample method '{upsample_method}'!"

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

            conv_upsample,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, kernel, 1, kernel // 2),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def build_generator(image_channels, image_size, model_channels, max_channels, latent_dim, skip_blocks, upsample_method,
                    dense_layers, kernel):
    upsamples = int(log2(image_size)) - 2

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
            blocks.append(nn.Linear(latent_dim, latent_dim))
            blocks.append(nn.BatchNorm1d(latent_dim))
            blocks.append(nn.LeakyReLU())

        init_channels = min((1 << upsamples) * model_channels, max_channels)
        blocks.append(nn.Linear(latent_dim, init_channels * 4 * 4))
        blocks.append(nn.Unflatten(1, (init_channels, 4, 4)))
        return blocks

    return nn.Sequential(
        *dense_block(),
        *res_block(),

        nn.BatchNorm2d(model_channels),
        nn.LeakyReLU(),

        nn.Conv2d(model_channels, image_channels, kernel, 1, kernel // 2),
        nn.Tanh(),
    )


def build_discriminator(image_channels, image_size, model_channels, max_channels, skip_blocks, downsample_method,
                        kernel):
    downsamples = int(log2(image_size)) - 2

    def res_block():
        blocks = []
        for index in range(downsamples):
            in_channels = min((1 << index) * 1 * model_channels, max_channels)
            out_channels = min((1 << index) * 2 * model_channels, max_channels)

            blocks.append(ResidualBlockDown(in_channels, out_channels, image_size >> index, downsample_method, kernel))

            for _ in range(skip_blocks - 1):
                blocks.append(ResidualBlockDown(out_channels, out_channels, image_size >> (index + 1), 'none', kernel))

        return blocks

    return nn.Sequential(
        nn.Conv2d(image_channels, model_channels, kernel, 1, kernel // 2),

        *res_block(),

        nn.Flatten(),
        nn.Linear(min((1 << downsamples) * model_channels, max_channels) * 4 * 4, 1),
    )


def build_standard_wgan_gp(config):
    image_channels = config['image_channels']
    image_size = config['image_size']
    latent_dim = config['latent_dim']

    learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
    beta1 = config['beta1'] if 'beta1' in config else 0.9
    beta2 = config['beta2'] if 'beta2' in config else 0.999

    upsample_method = config['upsample_method']
    dense_layers = config['dense_layers'] if 'dense_layers' in config else 1
    generator_model_channels = config['generator_model_channels']
    generator_max_channels = config['generator_max_channels']
    generator_skip_blocks = config['generator_skip_blocks'] if 'generator_skip_blocks' in config else 1
    generator_kernel = config['generator_kernel'] if 'generator_kernel' in config else 3
    generator = build_generator(image_channels, image_size, generator_model_channels,
                                generator_max_channels, latent_dim, generator_skip_blocks, upsample_method,
                                dense_layers, generator_kernel)

    downsample_method = config['downsample_method']
    discriminator_model_channels = config['discriminator_model_channels']
    discriminator_max_channels = config['discriminator_max_channels']
    discriminator_skip_blocks = config['discriminator_skip_blocks'] if 'discriminator_skip_blocks' in config else 1
    discriminator_kernel = config['discriminator_kernel'] if 'discriminator_kernel' in config else 3
    discriminator = build_discriminator(image_channels, image_size, discriminator_model_channels,
                                        discriminator_max_channels, discriminator_skip_blocks, downsample_method,
                                        discriminator_kernel)

    batch_size = config['batch_size']
    dataset = config['dataset']
    dataloader = numpy_dataloader(dataset, batch_size, 8)

    print_summary = config['print_summary'] if 'print_summary' in config else False
    gan = GanModelBase(dataloader, generator, discriminator, latent_dim,
                       lr=learning_rate, betas=(beta1, beta2), summary=print_summary)

    gan.batch_size = batch_size
    gan.gradient_updates = config['gradient_updates'] if 'gradient_updates' in config else 1
    gan.save_snapshot_rate = config['save_snapshot_rate'] if 'save_snapshot_rate' in config else 100
    gan.save_model_rate = config['save_model_rate'] if 'save_model_rate' in config else 1000
    gan.critic_updates = config['critic_updates'] if 'critic_updates' in config else 5
    gan.gradient_penalty_lambda = config['gradient_penalty_lambda'] if 'gradient_penalty_lambda' in config else 10

    return gan
