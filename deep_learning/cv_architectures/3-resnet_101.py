#!/usr/bin/env python3
"""Builds the ResNet-101 architecture as described in
'Deep Residual Learning for Image Recognition' (He et al., 2015).
"""

from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Builds a bottleneck residual block (1x1 -> 3x3 -> 1x1 convs)."""
    shortcut = x

    x = keras.layers.Conv2D(
        filters,
        1,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = keras.layers.ReLU(name=f"{name}_relu1")(x)

    x = keras.layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = keras.layers.ReLU(name=f"{name}_relu2")(x)

    x = keras.layers.Conv2D(
        filters * 4,
        1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv3",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn3")(x)

    if downsample:
        shortcut = keras.layers.Conv2D(
            filters * 4,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_downsample_conv",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(name=f"{name}_downsample_bn")(
            shortcut
        )

    x = keras.layers.Add(name=f"{name}_add")([x, shortcut])
    x = keras.layers.ReLU(name=f"{name}_out")(x)

    return model
