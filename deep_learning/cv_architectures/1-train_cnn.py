#!/usr/bin/env python3
"""function that trains CNN models."""

from tensorflow import keras


def compile_and_train_cnn(
    model,
    epochs,
    batch_size,
    x_train,
    y_train,
    x_val,
    y_val,
    optimizer_name="adam",
    optimizer_params=None,
):
    """Returns the trained CNN model, raining history object."""
    if optimizer_params is None:
        optimizer_params = {}
    optimizer_map = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    optimizer_class = optimizer_map[optimizer_name]
    optimizer = optimizer_class(**optimizer_params)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )
    return model, history
