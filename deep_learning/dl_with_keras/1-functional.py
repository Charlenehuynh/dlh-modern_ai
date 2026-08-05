#!/usr/bin/env python3
"""Build a shallow neural network using the Functional API."""

from tensorflow import keras


def build_model(input_dim, neurons_h):
    """
    Creates a shallow neural network using keras.Model (Functional API).

    Args:
        input_dim (int): number of input features.
        neurons_h (int): number of neurons in the hidden layer.

    Returns:
        model: a Keras Model.
    """
    inputs = keras.Input(shape=(input_dim,))
    hidden = keras.layers.Dense(neurons_h, activation='sigmoid')(inputs)
    outputs = keras.layers.Dense(10, activation='softmax')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
