from os import listdir, makedirs, remove
from os.path import join, isfile
from random import randint
from multiprocessing import Process
from math import ceil

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


def pre_generate_dataset_parallel(image_folders, transform, output_folder, sample_count, workers):
    samples_each = ceil(sample_count / workers)
    processes = []
    for i in range(workers):
        p = Process(target=pre_generate_dataset, args=(image_folders, transform, output_folder, samples_each, i * samples_each))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def pre_generate_dataset(image_folders, transform, output_folder, sample_count, offset=0):
    dataloader = image_dataloader(image_folders, 1, transform, workers=1)

    delete_file(output_folder)
    makedirs(output_folder, exist_ok=True)
    for i in tqdm(range(sample_count)):
        batch = next(dataloader)[0].cpu()
        save_batch(batch, output_folder, i + offset)


def save_batch(batch, output_folder, index):
    file = join(output_folder, str(index) + '.npy')
    np.save(file, batch.numpy())

class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator_func(**self.kwargs)


def load_dataset(folder, batch_size):
    files = [join(folder, file) for file in listdir(folder)]

    while True:
        batch = []
        for _ in range(batch_size):
            img_name = files[randint(0, len(files) - 1)]
            image = torch.from_numpy(np.load(img_name))
            batch.append(image)
        
        batch = torch.stack(batch)
        yield Variable(batch.type(FloatTensor))
