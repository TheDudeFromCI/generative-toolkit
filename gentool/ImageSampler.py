from math import sqrt

from .Noise import noise


def sample_image_from_noise(model, size, batch_size):
    z = noise(batch_size, size)
    return model(z), int(sqrt(batch_size))


def sample_image_to_image(model, batch):
    generated = model(batch)
    images = [val for pair in zip(batch, generated) for val in pair]
    return images, int(sqrt(len(batch))) * 2


def sample_image_to_image_masked(model, batch, modified):
    generated = model(modified)
    images = [val for pair in zip(batch, modified, generated) for val in pair]
    return images, int(sqrt(len(batch))) * 3
