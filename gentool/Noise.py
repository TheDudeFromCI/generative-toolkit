import numpy as np

import torch
from torch.cuda import FloatTensor


def noise(batch_size, size):
    return FloatTensor(np.random.normal(0, 1, (batch_size, *size)))


def noise_like(tensor):
    return FloatTensor(np.random.normal(0, 1, tensor.shape))


def from_mu_var(mu, var):
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return eps * std + mu


def instance_noise(batch, noise_strength):
    return batch + noise_like(batch) * noise_strength
