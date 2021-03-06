import torch
import torchinfo
import torch.autograd as autograd
from torch.nn import functional as F
from torch.autograd.variable import Variable

from ModelBase import ImageModelBase
from GanImageSwapper import GanImageSwapper
from AdaptiveDataAug import augment


class GanModelBase(ImageModelBase):
    def __init__(self, dataloader, generator, discriminator, summary=False):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader

        self.batch_size = 1
        self.latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else 0
        self.image_size = generator.image_size
        self.image_channels = generator.image_channels
        self.sample_noise = self.noise((64, self.latent_dim))

        self.swap_buffer = 512
        self.swap_chance = 0.5
        self.swapper = None  # So we can have a chance to modify settings before start

        self.ada_augment = torch.tensor([0.0, 0.0])
        self.ada_target = 0.6
        self.ada_length = 50000000
        self.ada_aug_step = self.ada_target / self.ada_length
        self.ada_p = 0

        self.critic_updates = 5
        self.gradient_penalty_lambda = 10

        if summary:
            self.summary()

    def summary(self):
        torchinfo.summary(self, (1, self.latent_dim))

        params = self.count_params()
        print(
            f"Loaded GAN with {params['generator']:,} generator params and {params['discriminator']:,} discriminator params.")

    def forward(self, x):
        return self.discriminator(self.generator(x))

    def sample_images(self):
        images = self.generator(self.sample_noise)
        return images, 8

    def wgan_gp_discriminator_loss(self, batch, noise):
        batch, _ = augment(batch, self.ada_p)
        d_loss_real = self.discriminator(batch)

        generated = self.generator(noise).detach()
        generated = self.swapper.swap(generated)
        generated, _ = augment(generated, self.ada_p)
        d_loss_fake = self.discriminator(generated)

        gradient_penalty = self.calculate_gradient_penalty(batch.data, generated.data)
        self.update_ada_p(d_loss_real)

        return d_loss_fake.mean() - d_loss_real.mean() + gradient_penalty

    def update_ada_p(self, d_loss_real):
        ada_aug_data = torch.tensor((torch.sign(d_loss_real).sum().item(), d_loss_real.shape[0]))
        self.ada_augment += ada_aug_data
        if self.ada_augment[1] > (self.batch_size * 4 - 1):
            authen_out_signs, num_outputs = self.ada_augment.tolist()
            r_t_stat = authen_out_signs / num_outputs
            sign = 1 if r_t_stat > self.ada_target else -1
            self.ada_p += sign * self.ada_aug_step * num_outputs
            self.ada_p = min(1.0, max(0.0, self.ada_p))
            self.ada_augment.mul_(0.0)

    def wgan_generator_loss(self, noise):
        generated = self.generator(noise)
        generated, _ = augment(generated, self.ada_p)
        return -self.discriminator(generated).mean()

    def calculate_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand((real_data.size(0), 1, 1, 1))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)

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
        # gradients = gradients.view(gradients.size(0), -1)
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * self.gradient_penalty_lambda

    def gan_bce_discriminator_loss(self, batch, noise):
        one = torch.ones((len(batch), 1))
        zero = torch.zeros((len(batch), 1))

        real_loss = F.binary_cross_entropy(self.discriminator(batch), one)
        fake_loss = F.binary_cross_entropy(self.discriminator(self.generator(noise)), zero)

        return real_loss + fake_loss

    def gan_bce_generator_loss(self, noise):
        one = torch.ones((len(noise), 1))
        return F.binary_cross_entropy(self.discriminator(self.generator(noise)), one)

    def random_latent(self):
        return self.noise((self.batch_size, self.latent_dim))

    def fit(self, *args, **kargs):
        if self.swapper is None:
            self.swapper = GanImageSwapper(self.generator, (self.batch_size, self.latent_dim),
                                           self.swap_buffer, swap_chance=self.swap_chance)

        super().fit(*args, **kargs)

    def train_batch(self):
        g_loss = 0
        d_loss = 0

        for _ in range(self.critic_updates):
            self.discriminator.optimizer.zero_grad()

            for _ in range(self.gradient_updates):
                batch = next(self.dataloader)
                noise = self.random_latent()

                loss = self.wgan_gp_discriminator_loss(batch, noise)
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
