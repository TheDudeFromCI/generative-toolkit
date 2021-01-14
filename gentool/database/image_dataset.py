from os import listdir
from os.path import join as join_path

from random import randint
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class ImageDataset(Dataset):
    def __init__(self, folders, transform, batch_size, format='RGB'):
        self.transform = transform
        self.format = format
        self.batch_size = batch_size
        self.files = []

        for folder in folders:
            for file in listdir(folder):
                self.files.append(join_path(folder, file))

    def __len__(self):
        return self.batch_size

    def __getitem__(self, _):
        while len(self.files) > 0:
            img_name = self.files[randint(0, len(self.files) - 1)]
            try:
                image = Image.open(img_name).convert(self.format)
                image = self.transform(image)
                return image
            except UnidentifiedImageError:
                # File cannot be loaded. Remove from file list
                self.files.remove(img_name)
