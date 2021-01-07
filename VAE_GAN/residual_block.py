from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, bias=True, kernel=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_channels = min(in_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel, 1, 1, bias=bias),
            # nn.GroupNorm(3, hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            activation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(3, out_channels),
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
