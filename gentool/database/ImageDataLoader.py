from os import listdir
from os.path import join as join_path

from random import randint
from PIL import Image, UnidentifiedImageError

import torch
from torch.cuda import FloatTensor
from torch.autograd.variable import Variable

from gentool.database.EmptyDatabaseError import EmptyDatabaseError


class ImageDataLoader:
    def __init__(self, folders, batch_size, transform, format='RGB'):
        self.batch_size = batch_size
        self.transform = transform
        self.format = format
        self.files = []

        for folder in folders:
            for file in listdir(folder):
                self.files.append(join_path(folder, file))

    def _random_file(self):
        while len(self.files) > 0:
            index = randint(0, len(self.files) - 1)
            img_name = self.files[index]

            try:
                image = Image.open(img_name).convert(self.format)
                return self.transform(image)
            except UnidentifiedImageError:
                # File cannot be loaded. Remove from file list
                self.files.remove(img_name)
                continue

        raise EmptyDatabaseError

    def __iter__(self):
        return self

    def __next__(self):
        samples = []
        for _ in range(self.batch_size):
            samples.append(self._random_file())

        tensor = torch.stack(samples)
        return Variable(tensor.type(FloatTensor))
