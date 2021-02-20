from math import sqrt

from GanModelBase import GanModelBase


class Img2ImgGan(GanModelBase):
    def __init__(self, dataloader_input, dataloader_output, generator, discriminator, summary=False):
        super().__init__(None, generator, discriminator, summary=summary)

        self.dataloader_output = dataloader_output
        self.dataloader_input = dataloader_input
        self.input_sample_images = next(self.dataloader_input)

    def sample_images(self):
        images = self.generator(self.input_sample_images)
        rows = int(sqrt(len(images))) * 3
        return images, rows

    def train_batch(self):
        g_loss = 0
        d_loss = 0

        for _ in range(self.critic_updates):
            self.discriminator.optimizer.zero_grad()

            for _ in range(self.gradient_updates):
                input_images = next(self.dataloader_input)
                output_images = next(self.dataloader_output)

                loss = self.wgan_gp_discriminator_loss(output_images, input_images)
                loss.backward()

                d_loss += loss.item() / self.gradient_updates / self.critic_updates

            self.discriminator.optimizer.step()

        self.generator.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            input_images = next(self.dataloader_input)

            loss = self.wgan_generator_loss(input_images)
            loss.backward()

            g_loss += loss.item() / self.gradient_updates

        self.generator.optimizer.step()

        return '[g_loss: {:.6f}, d_loss: {:.6f}]'.format(g_loss, d_loss)
