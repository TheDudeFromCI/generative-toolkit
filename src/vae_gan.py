from torch import nn
from torch.cuda import FloatTensor
from torch.optim import Adam
from torch.autograd import Variable

from src.vae import VAE
from src.encoder import Encoder
from src.decoder import Decoder


class VAE_GAN(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim, dataloader):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.dataloader = dataloader

        self.encoder = Encoder(image_size, image_channels,
                               layers_per_size=4)

        self.decoder = Decoder(image_size, image_channels, latent_dim,
                               nn.Sigmoid(), layers_per_size=4)

        self.vae = VAE(self.encoder, self.decoder, latent_dim)

    def train_vae(self, epochs=100):
        batch_count = len(self.dataloader)

        optimizer = Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))
                err = self.vae.train_batch(sample, optimizer)

                print('[Epoch {}/{}] (Batch {}/{}) Err: {:.4f}'.format(epoch + 1,
                                                                       epochs, batch_number + 1, batch_count, err))
