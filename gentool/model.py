import os
import math
import time
from livelossplot.outputs.bokeh_plot import BokehPlot
from livelossplot.plot_losses import PlotLosses

import torch
from torch import save
from torchvision.utils import save_image
from torch.autograd import Variable
from torchinfo import summary

from .vae_gan import VAE_GAN
from .image_dataset import ImageDataset

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST


class ModelParameters:
    def __init__(self):
        self.image_size = 256
        self.image_channels = 3
        self.layers_per_size = 5
        self.channel_scaling = 2
        self.latent_dim = 2048

        self.database = 'mnist'
        self.database_path = './data'
        self.batch_size = 16

        self.vae_learning_rate = 1e-3
        self.gan_learning_rate = 1e-3
        self.vae_pretraining_epochs = 50
        self.epochs = 500

        self.print_summary = True
        self.save_snapshots = True
        self.save_model = True
        self.plot_loss = True
        self.cuda = True
        self.epoch_callback = None
        self.print_info = True

    @property
    def total_layers(self):
        return int(math.log2(self.image_size)) * self.layers_per_size

    @property
    def max_channels(self):
        return self.image_channels * self.channel_scaling ** int(math.log2(self.image_size))


class Model:
    def __init__(self, parameters: ModelParameters):
        self.parameters = parameters
        self.dataloader = get_dataloader(parameters)

        groups = {
            'VAE': ['recons_loss'],
            'GAN': ['g_loss', 'd_loss']
        }
        self.liveloss = PlotLosses(
            outputs=[BokehPlot(max_cols=2)], mode='script', groups=groups)

        self.vae_gan = VAE_GAN(parameters.image_size, parameters.image_channels, parameters.latent_dim,
                               self.dataloader, layers_per_size=parameters.layers_per_size, channel_scaling=parameters.channel_scaling)
        self.vae_gan.eval()

        if parameters.cuda:
            self.vae_gan.cuda()

        if parameters.print_summary:
            print('Model Summary:')
            summary(self.vae_gan, depth=10)

    def _epoch_callback(self, epoch, recons_loss=0, kld_loss=0, g_loss=0, d_loss=0):
        self.recons_loss = recons_loss
        self.kld_loss = kld_loss
        self.g_loss = g_loss
        self.d_loss = d_loss

        if self.parameters.print_info:
            print('Finished epoch {} with a loss of {:.4f}.'.format(
                epoch, recons_loss))

        if self.parameters.save_snapshots:
            save_vae_snapshot(self.vae_gan.vae, self.dataloader,
                              epoch, self.parameters.cuda)
            save_gan_snapshot(self.vae_gan.gan, epoch)

        if self.parameters.save_model:
            save_model(self.vae_gan, epoch)

        if self.parameters.plot_loss:
            logs = {
                'recons_loss': recons_loss + kld_loss,
                'g_loss': g_loss,
                'd_loss': d_loss
            }

            self.liveloss.update(logs, current_step=epoch)
            self.liveloss.send()

        if self.parameters.epoch_callback:
            self.parameters.epoch_callback(
                epoch, recons_loss, kld_loss, g_loss, d_loss)

    def train(self):
        self.vae_gan.train()

        if self.parameters.vae_pretraining_epochs > 0:
            if self.parameters.print_info:
                print('Pre-training VAE.')

            self.vae_gan.train_vae(epochs=self.parameters.vae_pretraining_epochs,
                                   epoch_callback=self._epoch_callback, print_info=self.parameters.print_info)

        if self.parameters.print_info:
            print('Converting to dual-training.')

        self.vae_gan.train_dual(epochs=self.parameters.epochs, epoch_offset=self.parameters.vae_pretraining_epochs,
                                epoch_callback=self._epoch_callback, print_info=self.parameters.print_info)

        self.vae_gan.eval()


def save_vae_snapshot(vae, dataloader, epoch, cuda):
    os.makedirs('images', exist_ok=True)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    _, batch = next(enumerate(dataloader))
    batch = Variable(batch[0].type(Tensor))

    images = vae.sample_output(batch)
    file = 'images/vae_sample_ep-{}.png'.format(epoch)
    save_image(images, file, nrow=8)


def save_gan_snapshot(gan, epoch):
    os.makedirs('images', exist_ok=True)

    images = gan.g_sample(9)
    file = 'images/gan_sample_ep-{}.png'.format(epoch)
    save_image(images, file, nrow=3)


def save_model(model, epoch):
    os.makedirs('models', exist_ok=True)
    file = 'models/ep-{}_{}.pth.tar'.format(epoch, time.time())
    save(model, file)


def get_dataloader(parameters: ModelParameters):
    blocks = [T.RandomAffine(45), T.ToTensor()]

    image_size = parameters.image_size
    blocks.append(T.RandomResizedCrop(image_size, scale=(0.333, 1.0)))
    blocks.append(T.ColorJitter(0.1, 0.1, 0.2, 0.4))
    blocks.append(T.RandomHorizontalFlip())
    transform = T.Compose(blocks)

    if parameters.database == 'mnist':
        dataset = MNIST(parameters.database_path, train=True,
                        download=True, transform=transform)
    elif parameters.database == 'cifar10':
        dataset = CIFAR10(parameters.database_path, train=True,
                          download=True, transform=transform)
    elif parameters.database == 'image_folder':
        color = 'RGB' if parameters.image_channels == 3 else 'L'
        dataset = ImageDataset(parameters.database_path, transform, color)
    else:
        raise ValueError('Unknown database: ' + parameters.database)

    return DataLoader(dataset, batch_size=parameters.batch_size, shuffle=True,
                      drop_last=True, num_workers=8)
