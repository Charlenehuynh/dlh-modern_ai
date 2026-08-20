#!/usr/bin/env python3
from tensorflow import keras


def get_early_stopping_callback(patience, monitor="val_loss", verbose=1):
    callback = keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=verbose,
        restore_best_weights=True
    )
    return callback
