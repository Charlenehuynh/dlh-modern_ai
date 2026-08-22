#!/usr/bin/env python3
"""initialize a Keras Tuner for hyperparameter tuning."""

import keras_tuner


def initiate_tuner(
    tuner_type,
    build_model,
    seed,
    hyperband_iterations,
    max_trials,
    objective,
    x_train=None,
    y_train=None,
):
    """
    return: A Keras Tuner object (Hyperband, RandomSearch, BayesianOpt)
    """
    if tuner_type == "RandomSearch":
        tuner = keras_tuner.RandomSearch(
            build_model,
            objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=True,
        )
    elif tuner_type == "Hyperband":
        tuner = keras_tuner.Hyperband(
            build_model,
            objective,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            overwrite=True,
        )
    elif tuner_type == "BayesianOptimization":
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective,
            max_trials=max_trials,
            seed=seed,
            overwrite=True,
        )
    return tuner
