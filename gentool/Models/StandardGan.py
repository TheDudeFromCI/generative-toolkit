from math import log2

from torch import nn

from ModelBase import GanModelBase
from Database import numpy_dataloader


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, downsample):
        super().__init__()

        if downsample:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            )
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv = nn.Sequential(
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2, 2) if downsample else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsample):
        super().__init__()

        if upsample:
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1) if upsample else nn.Conv2d(
                in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def build_generator(image_channels, image_size, model_channels, max_channels, latent_dim, skip_blocks):
    upsamples = int(log2(image_size)) - 2

    def res_block():
        blocks = []
        for index in range(upsamples - 1, -1, -1):
            in_channels = min((1 << index) * 2 * model_channels, max_channels)
            out_channels = min((1 << index) * 1 * model_channels, max_channels)

            for _ in range(skip_blocks - 1):
                blocks.append(ResidualBlockUp(in_channels, in_channels, False))

            blocks.append(ResidualBlockUp(in_channels, out_channels, True))
        return blocks

    return nn.Sequential(
        nn.Linear(latent_dim, min((1 << upsamples) * model_channels, max_channels) * 4 * 4),
        nn.Unflatten(1, (min((1 << upsamples) * model_channels, max_channels), 4, 4)),

        *res_block(),

        nn.BatchNorm2d(model_channels),
        nn.LeakyReLU(),

        nn.Conv2d(model_channels, image_channels, 3, 1, 1),
        nn.Tanh(),
    )


def build_discriminator(image_channels, image_size, model_channels, max_channels, skip_blocks):
    downsamples = int(log2(image_size)) - 2

    def res_block():
        blocks = []
        for index in range(downsamples):
            in_channels = min((1 << index) * 1 * model_channels, max_channels)
            out_channels = min((1 << index) * 2 * model_channels, max_channels)

            blocks.append(ResidualBlockDown(in_channels, out_channels, image_size >> index, True))

            for _ in range(skip_blocks - 1):
                blocks.append(ResidualBlockDown(out_channels, out_channels, image_size >> (index + 1), False))

        return blocks

    return nn.Sequential(
        nn.Conv2d(image_channels, model_channels, 3, 1, 1),

        *res_block(),

        nn.Flatten(),
        nn.Linear(min((1 << downsamples) * model_channels, max_channels) * 4 * 4, 1),
    )


def build_standard_wgan_gp(config):
    image_channels = config['image_channels']
    image_size = config['image_size']
    model_channels = config['model_channels']
    max_channels = config['max_channels'] if 'max_channels' in config else 128
    latent_dim = config['latent_dim']
    skip_blocks = config['skip_blocks'] if 'skip_blocks' in config else 1

    learning_rate = config['learning_rate'] if 'learning_rate' in config else 1
    beta1 = config['beta1'] if 'beta1' in config else 0.9
    beta2 = config['beta2'] if 'beta2' in config else 0.999

    generator = build_generator(image_channels, image_size, model_channels, max_channels, latent_dim, skip_blocks)
    discriminator = build_discriminator(image_channels, image_size, model_channels, max_channels, skip_blocks)

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
