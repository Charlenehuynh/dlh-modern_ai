#!/usr/bin/env python3
"""Create Keras model with L2 regularization"""

from tensorflow import keras


def build_model_with_L2_regularization(
        input_dim, hidden_units, n_layers, lambda_l2):
    """Return : model: A Keras model with L2 reg."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for i in range(n_layers):
        layer = keras.layers.Dense(
            units=hidden_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.L2(l2=lambda_l2),
        )
        model.add(layer)
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model
