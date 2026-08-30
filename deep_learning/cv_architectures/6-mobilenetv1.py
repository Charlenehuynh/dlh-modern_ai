#!/usr/bin/env python3
"""Defines the mobilenet function"""
from tensorflow import keras as K
mobilenet_backbone = __import__('5-mobilenet_backbone').mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """
    Builds the MobileNetV1 architecture as described in
    'MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications' (2017).

    Args:
        input_shape: tuple representing the input image shape.
        num_classes: number of output classes.

    Returns:
        A Keras Model instance representing MobileNetV1.
    """
    init = K.initializers.HeNormal(seed=0)

    X = K.Input(shape=input_shape)

    # Backbone (initial conv + 13 depthwise separable blocks)
    Y = mobilenet_backbone(X)

    # Classification head
    Y = K.layers.GlobalAveragePooling2D()(Y)
    Y = K.layers.Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=init
    )(Y)

    model = K.Model(inputs=X, outputs=Y, name='MobileNetV1')

    return model