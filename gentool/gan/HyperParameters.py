class Gan2DHyperParameters():
    def __init__(self):
        self.gen_learning_rate = 1e-4
        self.dis_learning_rate = 4e-4
        self.batch_size = 25
        self.image_folder = ''
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.instance_noise = 0.1
        self.latent_dim = 512
        self.generator = None
        self.discriminator = None
        self.critic_updates = 5
        self.gradient_updates = 1
