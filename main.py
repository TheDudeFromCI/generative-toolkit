import os

from src.vae_gan import VAE_GAN
from src.image_dataset import ImageDataset

from torch.cuda import FloatTensor
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchinfo import summary
from livelossplot import PlotLosses
from livelossplot.outputs import BokehPlot

COLOR_IMAGE = True
IMAGE_SIZE = 32
BATCH_SIZE = 32
DATA_FOLDER = 'data'
LATENT_DIM = 512

IMAGE_FORMAT = 'RGB' if COLOR_IMAGE else 'L'
IMAGE_CHANNELS = 3 if COLOR_IMAGE else 1


def save_vae_snapshot(vae, dataloader, epoch):
    os.makedirs('images', exist_ok=True)

    _, batch = next(enumerate(dataloader))
    batch = Variable(batch[0].type(FloatTensor))

    images = vae.sample_output(batch)
    file = 'images/vae_sample_ep-{}.png'.format(epoch)
    save_image(images, file, nrow=8)


def save_gan_snapshot(gan, epoch):
    os.makedirs('images', exist_ok=True)

    images = gan.g_sample(25)
    file = 'images/gan_sample_ep-{}.png'.format(epoch)
    save_image(images, file, nrow=5)


def main():

    dataloader = DataLoader(
        CIFAR10(DATA_FOLDER, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(IMAGE_SIZE),
                ])),

        # If you want to train on your own images instead
        # ImageDataset(
        #     DATA_FOLDER,
        #     transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Resize(IMAGE_SIZE),
        #     ]),
        #     format=IMAGE_FORMAT
        # ),

        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    vae_gan = VAE_GAN(IMAGE_SIZE, IMAGE_CHANNELS, LATENT_DIM,
                      dataloader, layers_per_size=2, channel_scaling=3)
    vae_gan.cuda()

    print('Model Summary:')
    summary(vae_gan, depth=10)

    groups = {
        'VAE': ['recons_loss'],
        'GAN': ['g_loss', 'd_loss'],
        'Total': ['total']
    }
    liveloss = PlotLosses(
        outputs=[BokehPlot(max_cols=3)], mode='script', groups=groups)

    def epoch_callback(epoch, recons_loss, kld_loss, g_loss, d_loss):
        save_vae_snapshot(vae_gan.vae, dataloader, epoch)
        save_gan_snapshot(vae_gan.gan, epoch)

        logs = {
            'recons_loss': recons_loss + kld_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'total': recons_loss + kld_loss + g_loss + d_loss,
        }

        liveloss.update(logs, current_step=epoch)
        liveloss.send()

    vae_gan.train_dual(epochs=500,
                       epoch_callback=epoch_callback)


if __name__ == '__main__':
    main()
