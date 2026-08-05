#!/usr/bin/env python3
"""function that trains a Keras model."""


def train_model(model, X, Y, epochs, verbose=1):
    """
    model: Keras model.
    X: Input data, shape (number of examples, input features).
    Y: labels, shape (number of examples, 1).
    epochs: Number of training epochs.
    verbose: Verbosity mode (0 = silent, 1 = progress bar).
    """
    model.fit(X, Y, epochs=epochs, verbose=verbose)
