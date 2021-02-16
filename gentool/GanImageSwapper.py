from random import random, randint

import torch


class GanImageSwapper():
    def __init__(self, generator, noise_shape, buffer, swap_chance):
        self.buffer = buffer
        self.images = []
        self.noise = []
        self.swap_chance = swap_chance

        for _ in range(buffer):
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

        device1 = batch.device
        device2 = noise.device

        index = randint(0, self.buffer - 1)
        batch = batch.detach().cpu()
        noise = noise.detach().cpu()

        self.images[index], batch = batch, self.images[index]
        self.noise[index], noise = noise, self.noise[index]

        batch = batch.to(device1)
        noise = noise.to(device2)

        if noise is None:
            return batch
        else:
            return batch, noise
