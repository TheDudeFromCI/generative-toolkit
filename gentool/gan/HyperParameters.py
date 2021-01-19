from math import log2

from torchvision import transforms as T


class Gan2DHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 3
        self.discriminator_layers_per_size = 3
        self.gen_initial_channels = 32
        self.dis_initial_channels = 4
        self.learning_rate = 1e-4
        self.kernel = 5
        self.batch_size = 25
        self.image_folders = []
        self.data_augmentation = T.ToTensor()
        self.data_workers = 4
        self.dropout = 0.4
        self.normalization = 'group'
        self.adam_beta1 = 0.5
        self.adam_beta2 = 0.999
        self.bias_neurons = False
        self.leaky_relu_slope = 0.2
        self.input_noise = 0.1
        self.label_smoothing = 0.1

    @property
    def latent_dim(self):
        return (self.gen_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4

    @property
    def discriminator_out(self):
        return (self.dis_initial_channels << int(log2(self.image_size)))
