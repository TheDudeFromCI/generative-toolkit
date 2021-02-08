from math import log2


class Vae2DHyperParameters():
    def __init__(self):
        self.image_size = 32
        self.image_channels = 1
        self.learning_rate = 2e-4
        self.batch_size = 25
        self.image_folder = ''
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.kld_weight = 1
        self.gradient_updates = 1
        self.encoder = None
        self.decoder = None
        self.latent_dim = 256
