#!/usr/bin/env python3
"""function that return a compiled Keras model"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """return model: A Keras model with the described architecture."""
    if activation in ("sigmoid", "tanh"):
        initializer = keras.initializers.GlorotUniform()
        dense_activation = activation
        extra_layer = None
    elif activation == "relu":
        initializer = keras.initializers.HeNormal()
        dense_activation = "relu"
        extra_layer = None
    elif activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
        dense_activation = None  # apply LeakyReLU as its own layer
        extra_layer = keras.layers.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = [
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(
            hidden_units,
            activation=dense_activation,
            kernel_initializer=initializer,
        ),
    ]
    if extra_layer is not None:
        layers.append(extra_layer)
    layers.append(keras.layers.Dense(10, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
