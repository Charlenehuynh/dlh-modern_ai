#!/usr/bin/env python3
"""Builds the ResNet-101 architecture"""

from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """Builds a bottleneck residual block (1x1 -> 3x3 -> 1x1 convs).

    x: input tensor.
    filters: number of filters used for the first two convolutions;
             the final convolution uses filters * 4.
    stride: stride applied to the first convolution of the block (and,
            when downsample is True, to the projection shortcut).
    downsample: whether a 1x1 convolutional projection shortcut is
                needed (True for the first block of every stage).
    name: base name used to prefix every layer created in this block.

    Returns: output tensor of the block.
    """
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

    return x


def make_layer(x, blocks, filters, stride=1, name=None):
    """Stacks `blocks` bottleneck residual blocks into a single stage.

    The first block always performs the projection/downsample so that
    the shortcut's number of channels (and, when stride > 1, its
    spatial dimensions) matches the main path's output.
    """
    x = bottleneck_block(
        x, filters, stride=stride, downsample=True, name=f"{name}_block1"
    )
    for i in range(1, blocks):
        x = bottleneck_block(
            x, filters, stride=1, downsample=False, name=f"{name}_block{i + 1}"
        )
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """Builds the ResNet-101 architecture.

    input_shape: tuple representing the input image shape.
    num_classes: number of output classes.

    Returns: the Keras Model implementing ResNet-101.
    """
    input_layer = keras.Input(shape=input_shape, name="input_layer")

    # Initial conv + max pool (stem)
    x = keras.layers.Conv2D(
        64,
        7,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="conv1",
    )(input_layer)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same", name="maxpool"
    )(x)

    # Standard ResNet-101 stage configuration
    x = make_layer(x, blocks=3, filters=64, stride=1, name="layer1")
    x = make_layer(x, blocks=4, filters=128, stride=2, name="layer2")
    x = make_layer(x, blocks=23, filters=256, stride=2, name="layer3")
    x = make_layer(x, blocks=3, filters=512, stride=2, name="layer4")

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
    output = keras.layers.Dense(num_classes, activation="softmax", name="fc")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="resnet101")

    return model
