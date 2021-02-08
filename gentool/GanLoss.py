import numpy as np

import torch
import torch.autograd as autograd
from torch.cuda import FloatTensor
from torch.nn import functional as F
from torch.autograd.variable import Variable


def wgan_gp_discriminator_loss(generator, discriminator, batch, noise):
    d_loss_real = discriminator(batch).mean()

    generated = generator(noise)
    d_loss_fake = discriminator(generated).mean()

    gradient_penalty = calculate_gradient_penalty(batch.data, generated.data, discriminator)

    return d_loss_fake - d_loss_real + gradient_penalty


def wgan_generator_loss(generator, discriminator, noise):
    generated = generator(noise)
    return -discriminator(generated).mean()


def gan_bce_discriminator_loss(generator, discriminator, batch, noise):
    one = FloatTensor(np.ones((len(batch), 1)))
    zero = FloatTensor(np.zeros((len(batch), 1)))

    real_loss = F.binary_cross_entropy(discriminator(batch), one)
    fake_loss = F.binary_cross_entropy(discriminator(generator(noise)), zero)

    return real_loss + fake_loss


def gan_bce_generator_loss(generator, discriminator, noise):
    one = FloatTensor(np.ones((len(noise), 1)))
    return F.binary_cross_entropy(discriminator(generator(noise)), one)


def calculate_gradient_penalty(real_data, fake_data, discriminator):
    alpha = FloatTensor(np.random.random((real_data.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = Variable(FloatTensor(real_data.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty * 10
