import torchinfo
from torch.nn import functional as F

from ModelBase import ImageModelBase


class AeModelBase(ImageModelBase):
    def __init__(self, dataloader, encoder, decoder, summary=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader

        self.latent_dim = encoder.latent_dim
        self.image_size = encoder.image_size
        self.image_channels = encoder.image_channels
        self.sample = next(dataloader)

        self.cuda()

        if summary:
            torchinfo.summary(self, (1, self.image_channels, self.image_size, self.image_size))

            params = self.count_params()
            print(f"Loaded AE with {params['encoder']:,} encoder params and {params['decoder']:,} decoder params.")

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def sample_images(self):
        return self.sample_image_to_image(self.sample)

    def train_batch(self):
        ae_loss = 0

        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        for _ in range(self.gradient_updates):
            original = next(self.dataloader)
            generated = self(original)
            loss = F.mse_loss(generated, original)
            loss.backward()

            ae_loss += loss.item() / self.gradient_updates

        self.encoder.optimizer.step()
        self.decoder.optimizer.step()

        return f"loss: {ae_loss:.6f}"
