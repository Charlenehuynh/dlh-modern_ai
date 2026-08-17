#!/usr/bin/env python3
"""configured SGD-based optimizer based on the specified variant"""

from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """Returns:
    optimizer: A Gradient Descent optimizer.
    """

    if name == "SGD":
        momentum = 0.0
        nesterov = False
    elif name == "SGD+Momentum":
        nesterov = False
    elif name == "SGD+Momentum+Nesterov":
        nesterov = True
    optimizer = keras.optimizers.SGD(
        learning_rate=lr, momentum=momentum, nesterov=nesterov
    )
    return optimizer
