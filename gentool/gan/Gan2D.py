import numpy as np

import torch
from torch.optim import Adam
from torchinfo import summary
from torch.cuda import FloatTensor
from torch.autograd import Variable

from gentool.ModelBase import ImageModelBase
from gentool.gan.HyperParameters import Gan2DHyperParameters
from gentool.tools.PreGenDataset import load_dataset


class Gan2D(ImageModelBase):
    def __init__(self, hyper_parameters: Gan2DHyperParameters):
        super().__init__(load_dataset(hyper_parameters.image_folder, hyper_parameters.batch_size))

        self.hyper_parameters = hyper_parameters
        self.generator = self.hyper_parameters.generator
        self.discriminator = self.hyper_parameters.discriminator

        betas = (hyper_parameters.adam_beta1, hyper_parameters.adam_beta2)
        gen_lr, dis_lr = hyper_parameters.gen_learning_rate, hyper_parameters.dis_learning_rate
        self.generator_opt = Adam(self.generator.parameters(), lr=gen_lr, betas=betas)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=dis_lr, betas=betas)

        self.cuda()
        summary(self, (1, hyper_parameters.latent_dim), depth=3)

    def sample_images(self):
        z = self.z_noise()
        batch = next(self.dataloader)
        generated = torch.cat([self.generator(z), batch])
        return generated, 24

    def z_noise(self):
        batch_size = self.hyper_parameters.batch_size
        latent_dim = self.hyper_parameters.latent_dim
        return FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))

    def instance_noise(self, batch):
        return batch + torch.rand_like(batch) * self.hyper_parameters.instance_noise

    def train_batch(self):
        gradient_updates = self.hyper_parameters.gradient_updates
        critic_updates = self.hyper_parameters.critic_updates

        d_loss_avg = 0
        for _ in range(critic_updates):
            self.discriminator_opt.zero_grad()

            for _ in range(gradient_updates):
                batch = self.instance_noise(next(self.dataloader))
                d_loss_real = self.discriminator(batch)

                generated = self.instance_noise(self.generator(self.z_noise()).detach())
                d_loss_fake = self.discriminator(generated)

                gradient_penalty = self.calculate_gradient_penalty(batch, generated)

                d_loss = d_loss_fake.mean() - d_loss_real.mean() + gradient_penalty
                d_loss.backward()
                d_loss_avg += d_loss.item() / gradient_updates / critic_updates

                del batch, generated, d_loss_fake, d_loss_real, d_loss, gradient_penalty

            self.discriminator_opt.step()

        self.generator_opt.zero_grad()

        g_loss_avg = 0
        for _ in range(gradient_updates):
            generated = self.instance_noise(self.generator(self.z_noise()))
            g_loss = -self.discriminator(generated).mean()
            g_loss.backward()
            g_loss_avg += g_loss.item() / gradient_updates

        self.generator_opt.step()
        return '[g_loss: {:.4f}, d_loss: {:.4f}]'.format(g_loss_avg, d_loss_avg)

    def calculate_gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                        grad_outputs=FloatTensor(np.ones(prob_interpolated.size())),
                                        create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return 10 * ((gradients_norm - 1) ** 2).mean()

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
