#!/usr/bin/env python3
"""Builds a frozen feature extractor from a pretrained CNN"""

from tensorflow import keras


def build_feature_extractor():
    """Loads a pretrained MobileNetV2 model"""
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, outputs)
    return model
