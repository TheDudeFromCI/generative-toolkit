import optuna
from copy import copy

from .model import Model, ModelParameters


def optimize(parameters: ModelParameters, trial_count):
    def trial(trial):
        p2 = copy(parameters)
        p2.vae_pretraining_epochs = 1
        p2.epochs = 3
        p2.save_model = False
        p2.save_snapshots = False
        p2.print_summary = False
        p2.plot_loss = False
        p2.print_info = False

        p2.gan_learning_rate = trial.suggest_loguniform('gan_lr', 1e-5, 1e-2)
        p2.vae_learning_rate = trial.suggest_loguniform('vae_lr', 1e-5, 1e-2)
        p2.layers_per_size = trial.suggest_int('layers', 1, 4)
        p2.batch_size = trial.suggest_int('batch_size', 1, 22)
        p2.latent_dim = trial.suggest_int('latent', 32, 1600)

        def epoch_callback(epoch, recons_loss, kld_loss, g_loss, d_loss):
            trial.report(recons_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = Model(p2)
        model.epoch_callback = epoch_callback

        model.train()
        return model.recons_loss

    study = optuna.create_study()
    study.optimize(trial, n_trials=trial_count)

    best = study.best_params
    parameters.gan_learning_rate = best['gan_lr']
    parameters.vae_learning_rate = best['vae_lr']
    parameters.layers_per_size = best['layers']
    parameters.batch_size = best['batch_size']
    parameters.latent_dim = best['latent']

    return parameters
