from torch import nn
from torch.cuda import FloatTensor
from torch.optim import Adam
from torch.autograd import Variable

from src.vae import VAE
from src.gan import GAN
from src.encoder import Encoder
from src.decoder import Decoder


class VAE_GAN(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim, dataloader, layers_per_size=2, channel_scaling=2):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.dataloader = dataloader

        self.encoder = Encoder(image_size, image_channels,
                               layers_per_size=layers_per_size, channel_scaling=channel_scaling)

        self.decoder = Decoder(image_size, image_channels, latent_dim,
                               nn.Sigmoid(), layers_per_size=layers_per_size, channel_scaling=channel_scaling)

        self.vae = VAE(self.encoder, self.decoder, latent_dim)
        self.gan = GAN(self.encoder, self.decoder, latent_dim)

    def train_vae(self, epochs=100, epoch_callback=None):
        batch_count = len(self.dataloader)

        optimizer = Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(epochs):
            if epoch_callback is not None:
                epoch_callback(epoch)

            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))
                err = self.vae.train_batch(sample, optimizer)

                print('[Epoch {}/{}] (Batch {}/{}) Err: {:.4f}'.format(epoch + 1,
                                                                       epochs, batch_number + 1, batch_count, err))

    def train_gan(self, epochs=100, epoch_callback=None):
        batch_count = len(self.dataloader)

        for epoch in range(epochs):
            if epoch_callback is not None:
                epoch_callback(epoch)

            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))
                d_loss, g_loss = self.gan.train_batch(sample)

                print('[Epoch {}/{}] (Batch {}/{}) G_Loss: {:.4f}, D_Loss: {:.4f}'.format(epoch + 1,
                                                                                          epochs,
                                                                                          batch_number + 1,
                                                                                          batch_count,
                                                                                          g_loss,
                                                                                          d_loss))
