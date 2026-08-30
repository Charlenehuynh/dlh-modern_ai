#!/usr/bin/env python3
"""function that creates a CNN model."""

from tensorflow import keras


def create_cnn_model(
    input_shape, filters, kernel_sizes, activations, pooling_type="max"
):
    """ Returns a compiled CNN model. """
    layers = [keras.Input(shape=input_shape)]
    for f, k, a in zip(filters, kernel_sizes, activations):
        layers.append(keras.layers.Conv2D(f, k, activation=a))
        if pooling_type == "max":
            layers.append(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        elif pooling_type == "avg":
            layers.append(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dense(10, activation="softmax"))
    model = keras.Sequential(layers)
    return model
