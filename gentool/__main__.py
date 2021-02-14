import sys
import argparse

import torch

from ModelLoader import load_model


def main():
    parser = argparse.ArgumentParser(prog='gentool')
    parser.add_argument("--training", action='store_true', help="Whether or not to start the model in training mode.")
    parser.add_argument("--model", type=str, help="The model to loader.")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations to train for.")
    parser.add_argument("--itr_offset", type=int, default=0, help="Iteration count offset.")
    parser.add_argument("--no_cuda", action='store_false', help="Disables loading to GPU.")

    opt = parser.parse_args()

    if opt.model is None:
        print('Model not defined!')
        sys.exit(1)

    if opt.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    model = load_model(opt.model, opt.cuda)

    if opt.training:
        model.train()
    else:
        model.eval()

    if opt.training:
        model.fit(opt.iterations, offset=opt.itr_offset)


if __name__ == '__main__':
    main()
