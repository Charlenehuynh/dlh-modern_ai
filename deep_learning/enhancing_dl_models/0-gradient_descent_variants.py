#!/usr/bin/env python3

"""return a configured gradient descent optimizer"""

from tensorflow import keras


def train_with_gradient_descent_variant(variant, learning_rate, x_train,
                                        batch_size):
    """return a configured gradient descent optimizer"""
    if variant == "batch":
        batch_size = len(x_train)
    elif variant == "stochastic":
        batch_size = 1
    elif variant == "mini_batch":
        batch_size = batch_size
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
    )
    return optimizer, batch_size
