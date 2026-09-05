#!/usr/bin/env python3
"""Builds a data augmentation pipeline for training images."""

from tensorflow import keras

SEED = 42


def build_data_augmentation():
    """
    Creates a Keras Sequential model containing common image data
    augmentation operations, to be applied to training images before
    they are passed into the pretrained CNN.

    Returns:
        A tf.keras.Sequential model of augmentation layers, all
        seeded with 42 for reproducibility.
    """
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.15, seed=SEED),
            keras.layers.RandomZoom(0.15, seed=SEED),
            keras.layers.RandomContrast(0.1, seed=SEED),
        ]
    )

    return data_augmentation
