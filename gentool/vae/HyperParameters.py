from math import log2

from torchvision import transforms as T


class Vae2DHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 3
        self.encoder_layers_per_size = 3
        self.decoder_initial_channels = 4
        self.encoder_initial_channels = 4
        self.learning_rate = 2e-4
        self.kernel = 5
        self.batch_size = 25
        self.image_folders = []
        self.data_augmentation = T.ToTensor()
        self.data_workers = 4
        self.dropout = 0.2
        self.normalization = 'none'
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.bias_neurons = True
        self.leaky_relu_slope = 0.01
        self.kld_weight = 1

    @property
    def encoder_out(self):
        return (self.encoder_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4

    @property
    def latent_dim(self):
        return (self.decoder_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4
