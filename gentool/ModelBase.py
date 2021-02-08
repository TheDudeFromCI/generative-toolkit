import os
from tqdm import tqdm
from time import time
from abc import abstractmethod

import torch
from torch import nn
from torchvision.utils import save_image


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.save_model_rate = 1000

    def save_model(self, update_number):
        os.makedirs('models', exist_ok=True)
        filename = 'models/up_{}.{}.pth.tar'.format(update_number, time())
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)

    def fit(self, n_updates, offset=0):
        with tqdm(range(offset, n_updates + offset), smoothing=0.0) as prog_bar:
            losses = []
            for update_number in range(offset + 1, n_updates + offset + 1):
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
    def train_batch(self):
        raise NotImplementedError


class ImageModelBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.save_snapshot_rate = 100

    def batch_callback(self, update_number, losses):
        super().batch_callback(update_number, losses)

        if update_number % self.save_snapshot_rate == 0:
            self.save_snapshot(update_number=update_number)

    def save_snapshot(self, update_number=0):
        os.makedirs('images', exist_ok=True)

        self.train(False)
        with torch.no_grad():
            images, rows = self.sample_images()
        self.train(True)

        filename = 'images/up-{}.{}.png'.format(update_number, time())
        save_image(images, filename, nrow=rows)

    @abstractmethod
    def sample_images(_):
        raise NotImplementedError
