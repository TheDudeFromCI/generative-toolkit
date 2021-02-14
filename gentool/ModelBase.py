import os
from tqdm import tqdm
from time import time
from math import sqrt
from abc import abstractmethod

import torch
from torch import nn
from torchvision.utils import save_image


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.save_model_rate = 1000
        self.batch_size = 16
        self.gradient_updates = 1

    def save_model(self):
        for model in self.get_sub_models():
            model.save_model()

    def load_model(self):
        for model in self.get_sub_models():
            model.load_model()

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
            self.save_model()

    @abstractmethod
    def train_batch(_):
        raise NotImplementedError

    def noise(_, size):
        return torch.normal(0.0, 1.0, size=size)

    def get_sub_models(self):
        models = []

        for attribute in dir(self):
            model = getattr(self, attribute)
            if isinstance(model, nn.Module):
                models.append(model)

        return models

    def count_params(self):
        parameter_count = {}

        for attribute in dir(self):
            model = getattr(self, attribute)
            if not isinstance(model, nn.Module):
                continue

            parameter_count[attribute] = 0

            for param in model.parameters():
                parameter_count[attribute] += param.numel()

        return parameter_count


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

    def sample_image_from_noise(self, size):
        z = self.noise(size)
        return self(z), int(sqrt(self.batch_size))

    def sample_image_to_image(self, batch):
        generated = self(batch)
        images = [val for pair in zip(batch, generated) for val in pair]
        return images, int(sqrt(len(batch))) * 2

    def sample_image_to_image_masked(self, batch, modified):
        generated = self(modified)
        images = [val for pair in zip(batch, modified, generated) for val in pair]
        return images, int(sqrt(len(batch))) * 3

    def fit(self, n_updates, offset=0):
        self.save_snapshot(update_number=offset)
        super().fit(n_updates, offset)
