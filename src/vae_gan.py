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

        for epoch in range(epochs):
            if epoch_callback is not None:
                epoch_callback(epoch)

            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))
                recons_loss, kld_loss = self.vae.train_batch(sample)

                print('[Epoch {}/{}] (Batch {}/{}) Recon_Loss: {:.4f}, KLD_Loss: {:.4f}'
                      .format(epoch + 1, epochs, batch_number + 1, batch_count, recons_loss, kld_loss))

    def train_gan(self, epochs=100, epoch_callback=None):
        batch_count = len(self.dataloader)

        for epoch in range(epochs):
            if epoch_callback is not None:
                epoch_callback(epoch)

            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))
                d_loss, g_loss = self.gan.train_batch(sample)

                print('[Epoch {}/{}] (Batch {}/{}) G_Loss: {:.4f}, D_Loss: {:.4f}'
                      .format(epoch + 1, epochs, batch_number + 1, batch_count, g_loss, d_loss))

    def train_dual(self, epochs=100, epoch_callback=None):
        batch_count = len(self.dataloader)

        for epoch in range(epochs):
            recons_loss_total = 0
            kld_loss_total = 0
            g_loss_total = 0
            d_loss_total = 0

            for batch_number, sample in enumerate(self.dataloader):
                sample = Variable(sample[0].type(FloatTensor))

                recons_loss, kld_loss = self.vae.train_batch(sample)
                g_loss, d_loss = self.gan.train_batch(sample)

                recons_loss_total += recons_loss / batch_count
                kld_loss_total += kld_loss / batch_count
                g_loss_total += g_loss / batch_count
                d_loss_total += d_loss / batch_count

                print('[Epoch {}/{}] (Batch {}/{}) Recon_Loss: {:.4f}, KLD_Loss: {:.4f}, G_Loss: {:.4f}, D_Loss: {:.4f}, Total_Loss: {:.4f}'
                      .format(epoch + 1, epochs, batch_number + 1, batch_count, recons_loss, kld_loss, g_loss, d_loss,
                              recons_loss + kld_loss + g_loss + d_loss))

            if epoch_callback is not None:
                epoch_callback(epoch + 1, recons_loss_total,
                               kld_loss_total, g_loss_total, d_loss_total)
