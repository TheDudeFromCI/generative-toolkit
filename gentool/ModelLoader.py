from torch import nn

from .ModelBase import conv2d, SkipConnection


def load_network(network_design):
    layers = []

    for layer in network_design:
        layer_type = layer['type']

        if layer_type == 'unflatten':
            channels = layer['channels']
            image_size = layer['image_size']
            layers.append(nn.Unflatten(1, (channels, image_size, image_size)))

        elif layer_type == 'skip_conv':
            count = layer['count']
            channels = layer['channels']
            image_size = layer['image_size']
            kernel = layer['kernel']
            normalization = layer['normalization']
            activation = layer['activation']

            for _ in range(count):
                layers.append(SkipConnection(channels, image_size, kernel_size=kernel,
                                             normalization=normalization, activation=activation))

        elif layer_type == 'upsample':
            method = layer['method']

            if method == 'nearest':
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

            elif method == 'bilinear':
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))

            else:
                assert False, f"Unknown upsampling method '{method}'!"

        elif layer_type == 'conv':
            count = layer['count']
            in_channels = layer['channels']
            out_channels = layer['channels']
            image_size = layer['image_size']
            kernel = layer['kernel']
            normalization = layer['normalization']
            activation = layer['activation']

            for _ in range(count):
                layers.append(conv2d(in_channels, out_channels, image_size, kernel_size=kernel,
                                     normalization=normalization, activation=activation))

        elif layer_type == 'flatten':
            layers.append(nn.Flatten())

        elif layer_type == 'downsample_conv':
            in_channels = layer['channels']
            out_channels = layer['channels']
            out_image_size = layer['out_image_size']
            kernel = layer['kernel']
            normalization = layer['normalization']
            activation = layer['activation']

            layers.append(conv2d(in_channels, out_channels, out_image_size, kernel_size=kernel,
                                 normalization=normalization, activation=activation, downsample=True))

        else:
            assert False, f"Unknown layer type '{layer_type}'!"

    return nn.Sequential(*layers)
