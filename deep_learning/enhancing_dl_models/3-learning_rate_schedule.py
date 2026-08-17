#!/usr/bin/env python3
"""return a Keras SGD optimizer and a specified learning rate schedule."""

from tensorflow import keras


def get_optimizer_SGD_with_schedule(
    schedule_type, initial_lr, decay_steps, decay_rate, momentum
):
    """
        Arguments:

    schedule_type: (str) The schedule type: 'exponential' or 'inverse_time'.

    'exponential': Applies exponential decay to the learning rate.

    'inverse_time': Applies inverse time decay to the learning rate.

    initial_lr: (float) The initial learning rate.

    decay_steps: (int) The number of steps before applying decay.

    decay_rate: (float) the decay rate factor.

    momentum: (float) The momentum factor.

    The learning rate decay should occur in a stepwise fashion.
    """
    if schedule_type == "exponential":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True,
        )
    elif schedule_type == "inverse_time":
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True,
        )
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                     momentum=momentum)
    return optimizer, lr_schedule
