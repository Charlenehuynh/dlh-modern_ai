#!/usr/bin/env python3
"""Attaches a classification head to a pretrained feature extractor."""

from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """
    Attaches a custom classification head to a pretrained feature
    extractor.

    Args:
        base_model: A Keras Model whose output is a pooled feature
            vector.
        num_classes: An integer representing the number of output
            classes.

    Returns:
        A new Keras Model ready for classification.
    """
    x = base_model.output
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(base_model.input, outputs)
    return model
