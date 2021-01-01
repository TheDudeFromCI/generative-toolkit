import os

from src.vae_gan import VAE_GAN
from src.image_dataset import ImageDataset

from torch.cuda import FloatTensor
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

COLOR_IMAGE = True
IMAGE_SIZE = 32
BATCH_SIZE = 64
DATA_FOLDER = 'data'
LATENT_DIM = 64

IMAGE_FORMAT = 'RGB' if COLOR_IMAGE else 'L'
IMAGE_CHANNELS = 3 if COLOR_IMAGE else 1


def save_vae_snapshot(vae, dataloader):
    os.makedirs('images', exist_ok=True)

    _, batch = next(enumerate(dataloader))
    batch = Variable(batch[0].type(FloatTensor))

    images = vae.sample_output(batch)
    save_image(images, "images/vae_sample.png", nrow=8)


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

    vae_gan = VAE_GAN(IMAGE_SIZE, IMAGE_CHANNELS, LATENT_DIM, dataloader)
    vae_gan.cuda()

    vae_gan.train_vae(epochs=20)
    save_vae_snapshot(vae_gan.vae, dataloader)


if __name__ == '__main__':
    main()
