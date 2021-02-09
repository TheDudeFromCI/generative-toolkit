from ModelBase import GanModelBase
from Database import numpy_dataloader
from NetworkLoader import load_network


def build_standard_wgan_gp(config):
    generator = load_network(config['generator'])
    discriminator = load_network(config['discriminator'])

    batch_size = config['batch_size']
    dataset = config['dataset']
    dataloader = numpy_dataloader(dataset, batch_size, 8)

    learning_rate = config['learning_rate']
    betas = config['beta1'], config['beta2']

    latent_dim = config['latent_dim']
    print_summary = config['print_summary']

    gan = GanModelBase(dataloader, generator, discriminator, latent_dim,
                       lr=learning_rate, betas=betas, summary=print_summary)
    gan.batch_size = batch_size
    gan.gradient_updates = config['gradient_updates']
    gan.save_snapshot_rate = config['save_snapshot_rate']
    gan.save_model_rate = config['save_model_rate']
    gan.critic_updates = config['critic_updates']
    gan.gradient_penalty_lambda = config['gradient_penalty_lambda']

    return gan
