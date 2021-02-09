import json

from Models.StandardGan import build_standard_wgan_gp


def load_model(file):
    with open(file) as f:
        config = json.load(f)

    model_type = config['model_type']

    if model_type == 'wgan-gp':
        return build_standard_wgan_gp(config)

    assert False, f"Unknown model type '{model_type}'!"
