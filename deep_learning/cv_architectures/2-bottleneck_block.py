#!/usr/bin/env python3
"""that implements a ResNet bottleneck residual block."""

from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Builds a ResNet bottleneck residual block.

    Args:
        x (tensor): input tensor to the block.
        filters (int): number of filters for the bottleneck's 3x3 conv
            (output channels = filters * 4).
        stride (int): stride applied at the first conv, used to downsample
        spatial dimensions (height/width).
        downsample (bool): if True, applies a 1x1 conv + BatchNorm to the
        skip connection so its shape matches the main path.
        name (str, optional): prefix for naming the layers in this block.

    Returns:
        tensor: output of the bottleneck block, same "type" of tensor as x,
        but possibly different shape (channels/spatial size).
    """
    shortcut = x

    out = keras.layers.Conv2D(
        filters, kernel_size=1, strides=stride, use_bias=False,
        name=f"{name}_conv1"
    )(x)
    out = keras.layers.BatchNormalization(name=f"{name}_bn1")(out)
    out = keras.layers.ReLU(name=f"{name}_relu1")(out)

    out = keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(out)
    out = keras.layers.BatchNormalization(name=f"{name}_bn2")(out)
    out = keras.layers.ReLU(name=f"{name}_relu2")(out)

    out = keras.layers.Conv2D(
        filters * 4, kernel_size=1, strides=1, use_bias=False,
        name=f"{name}_conv3"
    )(out)
    out = keras.layers.BatchNormalization(name=f"{name}_bn3")(out)

    if downsample:
        shortcut = keras.layers.Conv2D(
            filters * 4,
            kernel_size=1,
            strides=stride,
            use_bias=False,
            name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    out = keras.layers.Add(name=f"{name}_add")([out, shortcut])
    out = keras.layers.ReLU(name=f"{name}_out_relu")(out)
    return out
