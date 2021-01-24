from os import listdir, makedirs, remove
from os.path import join, isfile
from random import randint

import numpy as np
from tqdm import tqdm

import torch
from torch.cuda import FloatTensor
from torch.autograd.variable import Variable

from gentool.database.ImageDataLoader import image_dataloader


def delete_file(file):
    if isfile(file):
        remove(file)
    else:
        for f in listdir(file):
            delete_file(join(file, f))

def pre_generate_dataset(image_folders, transform, output_folder, sample_count, batch_size):
    dataloader = image_dataloader(image_folders, batch_size,
                                  transform, workers=1)

    delete_file(output_folder)
    makedirs(output_folder, exist_ok=True)
    for i in tqdm(range(sample_count)):
        batch = next(dataloader).cpu()

        file = join(output_folder, str(i) + '.npy')
        np.save(file, batch.numpy())


def load_dataset(folder):
    files = [join(folder, file) for file in listdir(folder)]

    def loader():
        while True:
            img_name = files[randint(0, len(files) - 1)]
            batch = torch.from_numpy(np.load(img_name))
            yield Variable(batch.type(FloatTensor))

    return loader
