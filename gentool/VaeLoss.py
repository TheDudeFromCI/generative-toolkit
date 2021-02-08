from math import log

import torch


def vae_logcosh_loss(model, batch, kld_weight):
    alpha = 10
    beta = 1

    generated, mu, var = model(batch)

    t = generated - batch
    recon_loss = alpha * t + torch.log(1 + torch.exp(-2 * alpha * t)) - log(2)
    recon_loss = (1 / alpha) * recon_loss.mean()

    kld_loss = -0.5 * torch.mean(1 + var - mu ** 2 - var.exp()) * kld_weight * beta
    return recon_loss + kld_loss
