#!/usr/bin/env python3
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
    optimizer_map = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }
    model.compile(
        optimizer_map[optimizer_name],
        loss="categorical_crossentropy",
        optimizer_params=optimizer_params,
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
