#!/usr/bin/env python3
"""Build a shallow neural network for multi-class classification."""

from tensorflow import keras


def build_model(input_dim, neurons_h):
    """create a shallow neural network with a single hidden layer"""
    model = keras.Sequential()
    model.add(keras.layers.Dense(
            neurons_h,
            activation="sigmoid",
            input_shape=(input_dim,))
    )
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model
