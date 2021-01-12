from os.path import join as join_path
from os import listdir

import torch
from random import randint
from PIL import Image, UnidentifiedImageError

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
            index = randint(0, len(self.files))
            img_name = self.files[index]

            try:
                image = Image.open(img_name).convert(self.format)
                image = self.transform(image)
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
            samples.append(self._random_file)

        return torch.stack(samples)
