from torch import nn


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, downsample_method, kernel, normalization, activation):
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
            get_normalization(normalization, in_channels, image_size),
            get_activation(activation),

            nn.Conv2d(in_channels, in_channels, kernel, 1, kernel // 2),
            get_normalization(normalization, in_channels, image_size),
            get_activation(activation),

            conv_downsample,
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, upsample_method, kernel, normalization, activation):
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
            get_normalization(normalization, in_channels, image_size),
            get_activation(activation),

            conv_upsample,
            get_normalization(normalization, out_channels, image_size),
            get_activation(activation),

            nn.Conv2d(out_channels, out_channels, kernel, 1, kernel // 2),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, normalization, activation):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv = nn.Sequential(
            get_normalization(normalization, in_channels, image_size),
            get_activation(activation),

            nn.Conv2d(in_channels, out_channels, kernel, 1, kernel // 2),
            get_normalization(normalization, out_channels, image_size),
            get_activation(activation),

            nn.Conv2d(out_channels, out_channels, kernel, 1, kernel // 2),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


def get_normalization_1d(name, channels):
    flags = {}
    if name.find(';') >= 0:
        parts = name.split(';')
        name = parts[0].strip()
        flags = dict(item.split('=') for item in parts[1].strip().split(' '))

    if name == 'batch':
        return nn.BatchNorm1d(channels)

    if name == 'group':
        groups = int(flags['groups']) if 'groups' in flags else 16
        return nn.GroupNorm(groups, channels)

    if name == 'layer':
        return nn.LayerNorm((channels,))

    if name == 'instance':
        return nn.InstanceNorm1d(channels)

    if name == 'none':
        return nn.Identity()

    assert False


def get_normalization(name, channels, image_size):
    flags = {}
    if name.find(';') >= 0:
        parts = name.split(';')
        name = parts[0].strip()
        flags = dict(item.split('=') for item in parts[1].strip().split(' '))

    if name == 'batch':
        return nn.BatchNorm2d(channels)

    if name == 'group':
        groups = int(flags['groups']) if 'groups' in flags else 16
        return nn.GroupNorm(groups, channels)

    if name == 'layer':
        return nn.LayerNorm((channels, image_size, image_size))

    if name == 'instance':
        return nn.InstanceNorm2d(channels)

    if name == 'none':
        return nn.Identity()

    assert False


def get_activation(name):
    flags = {}
    if name.find(';') >= 0:
        parts = name.split(';')
        name = parts[0].strip()
        flags = dict(item.split('=') for item in parts[1].strip().split(' '))

    if name == 'relu':
        return nn.ReLU()

    if name == 'leaky_relu':
        slope = float(flags['slope']) if 'slope' in flags else 0.01
        return nn.LeakyReLU(negative_slope=slope)

    if name == 'sigmoid':
        return nn.Sigmoid()

    if name == 'tanh':
        return nn.Tanh()

    if name == 'softmax':
        return nn.Softmax()

    if name == 'none':
        return nn.Identity()

    assert False


def conv2d(in_channels, out_channels, image_size, kernel_size=3, normalization='group', activation='leaky_relu', downsample=False,
           transpose=False):
    assert not (downsample and transpose), 'Cannot downsample and transpose at the same time!'

    if transpose:
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2, 1)
    else:
        stride = 2 if downsample else 1
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)

    if normalization == 'spectral':
        conv = nn.utils.spectral_norm(conv)
        norm = nn.Identity()
    else:
        norm = get_normalization(normalization, out_channels, image_size)

    activ = get_activation(activation)

    return nn.Sequential(conv, norm, activ)
