#!/usr/bin/env python3
"""create a Keras model with dropout regularization"""

from tensorflow import keras


def build_model_with_dropout(
    input_dim, hidden_units, n_layers, dropout_rate_input, dropout_rate_hidden
):
    """
    Arguments:
    input_dim: (int) Number of input features.
    hidden_units: (int) Number of neurons in each hidden layer.
    n_layers: (int) Number of hidden layers to include.
    dropout_rate_input: (float) Dropout rate to apply after the input layer.
    dropout_rate_hidden: (float) Dropout rate to apply after each hidden layer.
    Returns:

    model: A Keras model instance with the described architecture.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    model.add(keras.layers.Dropout(dropout_rate_input))
    for i in range(n_layers):
        layer = keras.layers.Dense(units=hidden_units, activation="relu")
        model.add(layer)
        model.add(keras.layers.Dropout(dropout_rate_hidden))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model
