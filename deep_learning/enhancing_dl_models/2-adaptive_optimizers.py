#!/usr/bin/env python3

"""returns a Keras optimizer based on the specified optimizer name"""

from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """return optimizer instance"""
    if name == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum)
    elif name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    elif name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho)
    return optimizer
