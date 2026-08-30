#!/usr/bin/env python3
"""Defines the mobilenet_backbone function"""

from tensorflow import keras as K

depthwise_separable_conv = __import__(
    "4-depthwise_separable_conv"
).depthwise_separable_conv


def mobilenet_backbone(inputs):
    """
    Builds the feature extraction backbone of MobileNetV1.

    Args:
        inputs: input tensor to the network.

    Returns:
        The output tensor of the MobileNet backbone (before classification).
    """
    init = K.initializers.HeNormal(seed=0)

    # Initial standard convolution, stride 2
    X = K.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=init,
        use_bias=False,
    )(inputs)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)

    # Stack of depthwise separable convolution blocks
    # (filters, stride) following the original MobileNetV1 pattern
    layer_config = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    for filters, stride in layer_config:
        X = depthwise_separable_conv(X, filters=filters, stride=stride)

    return X
