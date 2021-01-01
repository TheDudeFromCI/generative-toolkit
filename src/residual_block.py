from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, bias=True, kernel=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, 1, 1, bias=bias),
            nn.BatchNorm2d(in_channels),
            activation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = activation

    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.in_channels == self.out_channels:
            x += res

        x = self.activation(x)
        return x
