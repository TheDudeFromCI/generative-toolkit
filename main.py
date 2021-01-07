from pprint import pprint


from VAE_GAN import Model, ModelParameters, optimize


def main():
    parameters = ModelParameters()

    parameters.database = 'mnist'
    parameters.image_size = 32
    parameters.image_channels = 1
    parameters.batch_size = 22

    parameters.vae_pretraining_epochs = 10
    parameters.epochs = 100

    print('Optimizing Parameters...')
    pprint(parameters)
    optimize(parameters, 100)

    print('Optimized Parameters:')
    pprint(parameters)

    model = Model(parameters)
    model.train()


if __name__ == '__main__':
    main()
