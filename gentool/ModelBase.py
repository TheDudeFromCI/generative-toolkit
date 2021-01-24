import os
import sys

from abc import abstractmethod
from livelossplot.plot_losses import PlotLosses

from tqdm import tqdm
from time import time

import torch
from torch import nn
from torchvision.utils import save_image


class ModelBase(nn.Module):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader
        self.save_model_rate = 1000

    def save_model(self, update_number):
        os.makedirs('models', exist_ok=True)
        filename = 'models/up_{}.{}.pth.tar'.format(update_number, time())
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)

    def fit(self, n_updates):
        with tqdm(range(n_updates), smoothing=1.0) as prog_bar:
            losses = []
            for update_number in range(1, n_updates + 1):
                losses = self.train_batch()

                prog_bar.write('Batch = {}, Losses = {}'.format(update_number, losses))
                self.batch_callback(update_number, losses)
                prog_bar.update(1)

    def batch_callback(self, update_number, losses):
        if update_number % self.save_model_rate == 0:
            self.save_model(update_number)

    def format_logs(self, losses):
        logs = {}

        for i in range(len(losses)):
            logs[self.loss_names[i]] = losses[i]

        return logs

    @abstractmethod
    def train_batch(self, batch):
        raise NotImplementedError


class ImageModelBase(ModelBase):
    def __init__(self, dataloader):
        super().__init__(dataloader)

        self.save_snapshot_rate = 100
        self.save_snapshot_count = 25

    def batch_callback(self, update_number, losses):
        super().batch_callback(update_number, losses)

        if update_number % self.save_snapshot_rate == 0:
            self.save_snapshot(update_number)

    def save_snapshot(self, update_number):
        os.makedirs('images', exist_ok=True)

        self.train(False)
        with torch.no_grad():
            images, rows = self.sample_images(self.save_snapshot_count)
        self.train(True)

        filename = 'images/up-{}.{}.png'.format(update_number, time())
        save_image(images, filename, nrow=rows)

    @ abstractmethod
    def sample_images(self, count):
        raise NotImplementedError
