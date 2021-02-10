from math import log2

from torch import nn

from ModelBase import GanModelBase
from Database import numpy_dataloader


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, image_size):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )

        self.conv = nn.Sequential(
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.LayerNorm((in_channels, image_size, image_size)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def build_generator(image_channels, image_size, model_channels, max_channels, latent_dim):
    upsamples = int(log2(image_size)) - 2

    return nn.Sequential(
        nn.Linear(latent_dim, min((1 << upsamples) * model_channels, max_channels) * 4 * 4),
        nn.Unflatten(1, (min((1 << upsamples) * model_channels, max_channels), 4, 4)),

        *[ResidualBlockUp(min((1 << i) * 2 * model_channels, max_channels),
                          min((1 << i) * 1 * model_channels, max_channels))
            for i in range(upsamples - 1, -1, -1)],

        nn.BatchNorm2d(model_channels),
        nn.LeakyReLU(),

        nn.Conv2d(model_channels, image_channels, 3, 1, 1),
        nn.Tanh(),
    )


def build_discriminator(image_channels, image_size, model_channels, max_channels):
    downsamples = int(log2(image_size)) - 2

    return nn.Sequential(
        nn.Conv2d(image_channels, model_channels, 3, 1, 1),

        *[ResidualBlockDown(min((1 << i) * 1 * model_channels, max_channels),
                            min((1 << i) * 2 * model_channels, max_channels),
                            image_size >> i)
            for i in range(downsamples)],

        nn.Flatten(),
        nn.Linear(min((1 << downsamples) * model_channels, max_channels) * 4 * 4, 1),
    )


def build_standard_wgan_gp(config):
    image_channels = config['image_channels']
    image_size = config['image_size']
    model_channels = config['model_channels']
    max_channels = config['max_channels']
    latent_dim = config['latent_dim']

    learning_rate = config['learning_rate']
    betas = config['beta1'], config['beta2']

    generator = build_generator(image_channels, image_size, model_channels, max_channels, latent_dim)
    discriminator = build_discriminator(image_channels, image_size, model_channels, max_channels)

    batch_size = config['batch_size']
    dataset = config['dataset']
    dataloader = numpy_dataloader(dataset, batch_size, 8)

    gan = GanModelBase(dataloader, generator, discriminator, latent_dim,
                       lr=learning_rate, betas=betas, summary=config['print_summary'])
    gan.batch_size = batch_size
    gan.gradient_updates = config['gradient_updates']
    gan.save_snapshot_rate = config['save_snapshot_rate']
    gan.save_model_rate = config['save_model_rate']
    gan.critic_updates = config['critic_updates']
    gan.gradient_penalty_lambda = config['gradient_penalty_lambda']

    return gan
