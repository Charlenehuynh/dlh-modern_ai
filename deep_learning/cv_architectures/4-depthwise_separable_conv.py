#!/usr/bin/env python3
"""Defines the depthwise_separable_conv function"""
from tensorflow import keras as K


def depthwise_separable_conv(X, filters, stride=1):
    """
    Builds a depthwise separable convolution block as used in MobileNetV1.

    Args:
        X: input tensor.
        filters: number of output channels for the pointwise convolution.
        stride: stride applied to the depthwise convolution.

    Returns:
        The output tensor of the depthwise separable convolution block.
    """
    init = K.initializers.HeNormal(seed=0)

    # Depthwise convolution
    depthwise = K.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        depthwise_initializer=init,
        use_bias=False
    )(X)
    depthwise = K.layers.BatchNormalization()(depthwise)
    depthwise = K.layers.ReLU()(depthwise)

    # Pointwise convolution
    pointwise = K.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        use_bias=False
    )(depthwise)
    pointwise = K.layers.BatchNormalization()(pointwise)
    pointwise = K.layers.ReLU()(pointwise)

    return pointwise