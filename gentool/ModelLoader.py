import json

from GanModelBase import GanModelBase
from Database import numpy_dataloader
from Models.Encoder import Encoder
from Models.Decoder import Decoder


def load_model(file):
    with open(file) as f:
        config = json.load(f)

    model_type = config['model_type']

    if model_type == 'wgan-gp':
        return build_standard_wgan_gp(config)

    if model_type == 'encoder':
        return Encoder(config)

    if model_type == 'decoder':
        return Decoder(config)

    assert False, f"Unknown model type '{model_type}'!"


def build_standard_wgan_gp(config):
    generator = load_model(config['generator'])
    discriminator = load_model(config['discriminator'])

    batch_size = config['batch_size']
    dataset = config['dataset']
    dataloader = numpy_dataloader(dataset, batch_size, 8)

    print_summary = config['print_summary'] if 'print_summary' in config else False
    gan = GanModelBase(dataloader, generator, discriminator, summary=print_summary)

    gan.batch_size = batch_size
    gan.gradient_updates = config['gradient_updates'] if 'gradient_updates' in config else 1
    gan.save_snapshot_rate = config['save_snapshot_rate'] if 'save_snapshot_rate' in config else 100
    gan.save_model_rate = config['save_model_rate'] if 'save_model_rate' in config else 1000
    gan.critic_updates = config['critic_updates'] if 'critic_updates' in config else 5
    gan.gradient_penalty_lambda = config['gradient_penalty_lambda'] if 'gradient_penalty_lambda' in config else 10

    return gan
