from src.model import Model, ModelParameters


def main():
    parameters = ModelParameters()

    parameters.database = 'mnist'
    parameters.image_size = 32
    parameters.image_channels = 1
    parameters.batch_size = 22

    parameters.vae_pretraining_epochs = 10
    parameters.epochs = 100

    model = Model(parameters)
    model.train()


if __name__ == '__main__':
    main()
