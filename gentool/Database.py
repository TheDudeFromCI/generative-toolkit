from os import listdir, makedirs, remove
from os.path import join, isfile

from math import ceil
from random import randint
from itertools import repeat

import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from PIL import Image, UnidentifiedImageError

import torch
from torch.cuda import FloatTensor
from torch.utils.data import Dataset
from torch.autograd.variable import Variable
from torch.utils.data.dataloader import DataLoader


def delete_file(file):
    if isfile(file):
        remove(file)
    else:
        for f in listdir(file):
            delete_file(join(file, f))


def get_files(image_folders):
    files = []
    for folder in image_folders:
        for file in listdir(folder):
            files.append(join(folder, file))

    return files


def pre_generate_dataset_parallel(image_folders, transform, output_folder, sample_count, workers, format='RGB', offset=0):
    delete_file(output_folder)
    makedirs(output_folder, exist_ok=True)
    files = get_files(image_folders)
    samples_each = ceil(sample_count / workers)

    def pre_generate_dataset(offset):
        for i in tqdm(range(sample_count)):
            while True:
                try:
                    img_name = files[randint(0, len(files) - 1)]
                    image = Image.open(img_name).convert(format)
                    image = transform(image)
                    save_batch(image, output_folder, i + offset)
                    break
                except UnidentifiedImageError:
                    continue

    processes = []
    for i in range(workers):
        p = Process(target=pre_generate_dataset, args=(i * samples_each + offset,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def pre_generate_dataset_1for1(image_folders, transform, output_folder, format='RGB', offset=0):
    delete_file(output_folder)
    makedirs(output_folder, exist_ok=True)

    files = get_files(image_folders)

    i = -1
    for img_name in tqdm(files):
        i += 1
        try:
            image = Image.open(img_name).convert(format)
            image = transform(image)
            save_batch(image, output_folder, i + offset)
        except UnidentifiedImageError:
            continue


def save_batch(batch, output_folder, index):
    file = join(output_folder, str(index) + '.npy')
    np.save(file, batch.numpy())


def numpy_dataloader(folder, batch_size, workers):
    dataloader = DataLoader(NumpyDataLoader(folder), batch_size=batch_size, num_workers=workers,
                            persistent_workers=True, prefetch_factor=8, drop_last=True)

    for loader in repeat(dataloader):
        for _, batch in enumerate(loader):
            batch = Variable(batch.type(FloatTensor))
            yield batch


class NumpyDataLoader(Dataset):
    def __init__(self, data_folder):
        self.files = [join(data_folder, f) for f in listdir(data_folder)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.from_numpy(np.load(self.files[index]))
