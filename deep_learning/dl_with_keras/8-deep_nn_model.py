#!/usr/bin/env python3
"""function to create a deep neural network"""

from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """return Keras mode"""
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for neurons in hidden_layers:
        model.add(keras.layers.Dense(neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model
