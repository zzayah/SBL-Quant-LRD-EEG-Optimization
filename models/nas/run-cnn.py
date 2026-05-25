import optuna

from cnn import N_TRIALS, objective


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
