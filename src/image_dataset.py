import os

import torch
from torch.utils.data import Dataset

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, dir, transform, format='RGB'):
        self.train_dir = dir + '/train'
        self.mask_dir = dir + '/mask'
        self.transform = transform
        self.format = format
        self.files = os.listdir(self.train_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.train_dir, self.files[index])
        image = Image.open(img_name).convert(self.format)
        image = self.transform(image)

        return [image]
