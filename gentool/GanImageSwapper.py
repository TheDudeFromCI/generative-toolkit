from random import random, randint

import torch
from tqdm import tqdm


class GanImageSwapper():
    def __init__(self, generator, noise_shape, buffer, swap_chance):
        self.buffer = buffer
        self.images = []
        self.noise = []
        self.swap_chance = swap_chance

        print('Generating GAN swap buffer...')
        for _ in tqdm(range(buffer)):
            noise = torch.normal(0.0, 1.0, size=noise_shape)
            batch = generator(noise).detach().cpu()
            self.images.append(batch)
            self.noise.append(noise.detach().cpu())

    def swap(self, batch, noise=None):
        if random() >= self.swap_chance:
            if noise is None:
                return batch
            else:
                return batch, noise

        index = randint(0, self.buffer - 1)

        device1 = batch.device
        batch = batch.detach().cpu()
        self.images[index], batch = batch, self.images[index]
        batch = batch.to(device1)

        if noise is not None:
            device2 = noise.device if noise is not None else None
            noise = noise.detach().cpu()
            self.noise[index], noise = noise, self.noise[index]
            noise = noise.to(device2)

        if noise is None:
            return batch
        else:
            return batch, noise
