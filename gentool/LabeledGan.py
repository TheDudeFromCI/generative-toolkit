import torch
from torch import autograd
from torch.nn import functional as F
from torch.autograd.variable import Variable

from GanModelBase import GanModelBase


class LabeledGan(GanModelBase):
    def __init__(self, dataloader, generator, discriminator, summary=False):
        super().__init__(dataloader, generator, discriminator, summary=summary)

    def wgan_gp_discriminator_loss(self, batch, labels, noise):
        d_loss_real, d_latent_real = self.discriminator(batch)
        d_loss_real = d_loss_real.mean()

        generated = self.generator(noise).detach()
        d_loss_fake, d_latent_fake = self.discriminator(generated)
        d_loss_fake = d_loss_fake.mean()

        gradient_penalty = self.calculate_gradient_penalty(batch.data, generated.data)

        latent_loss_real = F.mse_loss(d_latent_real, labels)
        latent_loss_fake = F.mse_loss(d_latent_fake, noise)
        latent_loss = latent_loss_real + latent_loss_fake

        return d_loss_fake - d_loss_real + gradient_penalty + latent_loss

    def calculate_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand((real_data.size(0), 1, 1, 1))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates, _ = self.discriminator(interpolates)
        fake = Variable(torch.ones((real_data.shape[0], 1)), requires_grad=False)
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
        return gradient_penalty * self.gradient_penalty_lambda

    def wgan_generator_loss(self, noise):
        generated = self.generator(noise)
        d_loss_fake, d_latent_fake = self.discriminator(generated)
        d_loss_fake = d_loss_fake.mean()

        latent_loss = F.mse_loss(d_latent_fake, noise)

        return -d_loss_fake + latent_loss

    def train_batch(self):
        g_loss = 0
        d_loss = 0

        for _ in range(self.critic_updates):
            self.discriminator.optimizer.zero_grad()

            for _ in range(self.gradient_updates):
                batch, labels = next(self.dataloader)
                noise = self.random_latent()

                loss = self.wgan_gp_discriminator_loss(batch, labels, noise)
                loss.backward()

                d_loss += loss.item() / self.gradient_updates / self.critic_updates

            self.discriminator.optimizer.step()

        self.generator.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            noise = self.random_latent()

            loss = self.wgan_generator_loss(noise)
            loss.backward()

            g_loss += loss.item() / self.gradient_updates

        self.generator.optimizer.step()

        return '[g_loss: {:.6f}, d_loss: {:.6f}]'.format(g_loss, d_loss)
