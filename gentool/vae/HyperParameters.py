from math import log2


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
        self.image_folder = ''
        self.dropout = 0
        self.normalization = 'batch'
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.kld_weight = 1

    @property
    def encoder_out(self):
        return (self.encoder_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4

    @property
    def latent_dim(self):
        return (self.decoder_initial_channels << int(log2(self.image_size) - 2)) * 4 * 4
