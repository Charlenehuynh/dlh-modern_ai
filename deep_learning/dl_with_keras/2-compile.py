#!/usr/bin/env python3
"""configure the keras model for training"""

from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """
    model: keras model.
    learning_rate: Learning rate for gradient descent (default is 0.01).
    """
    sgd_optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(
        optimizer=sgd_optimizer, loss="binary_crossentropy",
        metrics=["accuracy"]
    )
