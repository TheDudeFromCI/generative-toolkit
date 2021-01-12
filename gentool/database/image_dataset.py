from os import listdir
from os.path import join as join_path

from random import randint
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class ImageDataset(Dataset):
    def __init__(self, folders, transform, format='RGB'):
        self.transform = transform
        self.format = format
        self.files = []

        for folder in folders:
            for file in listdir(folder):
                self.files.append(join_path(folder, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]

        try:
            image = Image.open(img_name).convert(self.format)
            image = self.transform(image)
            return image
        except UnidentifiedImageError:
            # File cannot be loaded. Remove from file list
            self.files.remove(img_name)
            return self.__getitem__(randint(0, len(self.files) - 1))
