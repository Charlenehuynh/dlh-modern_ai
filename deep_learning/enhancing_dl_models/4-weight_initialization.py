#!/usr/bin/env python3
"""function that return a compiled Keras model"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    if activation == "sigmoid" or activation == "tanh":
        initializer = keras.initializers.GlorotUniform()
        hidden_activation = activation
    elif activation == "relu":
        initializer = keras.initializers.HeNormal()
        hidden_activation = activation
    if activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
        hidden_activation = keras.layers.LeakyReLU()

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(
                hidden_units,
                activation=hidden_activation,
                kernel_initializer=initializer,
            ),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model
