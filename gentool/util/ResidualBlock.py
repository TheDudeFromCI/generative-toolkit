from math import floor, log2
from torch import nn


def best_group_count(channels):
    target_channels = min(16, int(channels / 3))
    target_dist = floor(log2(target_channels) + 0.5)

    best_count = 1
    best_dist = target_dist

    for i in range(1, channels + 1):
        if channels % i != 0:
            continue

        dist = abs(floor(log2(i) + 0.5) - target_dist)
        if dist < best_dist:
            best_count = i
            best_dist = dist

    return int(best_count)


def get_normalization(norm_name, channels, image_size, learnable_params):
    if norm_name == 'none':
        return nn.Identity()

    if norm_name == 'batch':
        return nn.BatchNorm2d(channels)

    if norm_name == 'group':
        groups = best_group_count(channels)
        return nn.GroupNorm(groups, channels)

    if norm_name == 'layer':
        return nn.LayerNorm((channels, image_size, image_size), elementwise_affine=learnable_params)

    if norm_name == 'instance':
        return nn.InstanceNorm2d(channels, affine=learnable_params)

    raise ValueError("Normalization '{}' not found!".format(norm_name))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, activation=nn.LeakyReLU(), bias=True, kernel=3,
                 normalization='batch', learnable_params=True, skip_connections=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connections = skip_connections and in_channels == out_channels

        hidden_channels = min(in_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel, 1, 1, bias=bias),
            get_normalization(normalization, hidden_channels, image_size, learnable_params),
            activation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel, 1, 1, bias=bias),
            get_normalization(normalization, out_channels, image_size, learnable_params),
        )

        self.activation = activation

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.skip_connections:
            x += res

        x = self.activation(x)
        return x
