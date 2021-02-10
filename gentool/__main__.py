import sys
import argparse

from ModelLoader import load_model


def main():
    parser = argparse.ArgumentParser(prog='gentool')
    parser.add_argument("--training", action='store_true', help="Whether or not to start the model in training mode.")
    parser.add_argument("--model", type=str, help="The model to loader.")
    parser.add_argument("--checkpoint", type=str, help="To load a specific model checkpoint")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations to train for.")
    parser.add_argument("--itr_offset", type=int, default=0, help="Iteration count offset.")

    opt = parser.parse_args()

    if opt.model is None:
        print('Model not defined!')
        sys.exit(1)

    model = load_model(opt.model)

    if opt.checkpoint is not None:
        model.load_model(opt.checkpoint)

    if opt.training:
        model.train()
    else:
        model.eval()

    if opt.training:
        model.fit(opt.iterations, offset=opt.itr_offset)


if __name__ == '__main__':
    main()