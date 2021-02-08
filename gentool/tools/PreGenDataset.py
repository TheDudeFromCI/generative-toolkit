from time import sleep
from os import listdir, makedirs, remove
from os.path import join, isfile
from random import randint
from multiprocessing import Process, Queue
from math import ceil

import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

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
        p = Process(target=pre_generate_dataset, args=(image_folders,
                                                       transform, output_folder, samples_each, i * samples_each))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def pre_generate_dataset_1for1(image_folders, transform, output_folder, format='RGB', offset=0):
    files = []
    for folder in image_folders:
        for file in listdir(folder):
            files.append(join(folder, file))

    i = -1
    for img_name in tqdm(files):
        i += 1
        try:
            image = Image.open(img_name).convert(format)
            image = transform(image)
            save_batch(image, output_folder, i + offset)
        except UnidentifiedImageError:
            continue


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


def load_batch(q, files, batch_size):
    while True:
        while q.qsize() >= 16:
            sleep(0.002)

        batch = []
        for _ in range(batch_size):
            img_name = files[randint(0, len(files) - 1)]
            batch.append(np.load(img_name))

        q.put(np.stack(batch))


def load_dataset(folder, batch_size):
    files = [join(folder, file) for file in listdir(folder)]

    q = Queue()

    for _ in range(4):
        p = Process(target=load_batch, args=(q, files, batch_size))
        p.daemon = True
        p.start()

    while True:
        batch = torch.from_numpy(q.get())
        yield batch.type(FloatTensor)
