#!/usr/bin/env python3
"""create Keras model where parameters are tuned via Keras tuner"""

from tensorflow import keras


def build_model(hp):
    """
    Input Layer:
    A compiled Keras Sequential model
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(784,)))
    n_layers = hp.Int("num_layers", min_value=1, max_value=2, step=1)
    for i in range(n_layers):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units", min_value=4, max_value=12, step=4),
                activation=hp.Choice("activation", values=["relu", "sigmoid"]),
            )
        )
    model.add(keras.layers.Dense(10, activation="softmax"))
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3])
    optimiser = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimiser, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
