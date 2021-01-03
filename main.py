import os

from src.vae_gan import VAE_GAN
from src.image_dataset import ImageDataset

from torch.cuda import FloatTensor
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

COLOR_IMAGE = False
IMAGE_SIZE = 32
BATCH_SIZE = 64
DATA_FOLDER = 'data'
LATENT_DIM = 16

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
        MNIST(DATA_FOLDER, train=True, download=True,
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

    def epoch_callback(epoch):
        # save_vae_snapshot(vae_gan.vae, dataloader, epoch)
        save_gan_snapshot(vae_gan.gan, epoch)

    vae_gan.train_gan(epochs=40, epoch_callback=epoch_callback)


if __name__ == '__main__':
    main()
