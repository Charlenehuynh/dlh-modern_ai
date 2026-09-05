#!/usr/bin/env python3
"""Unfreezes the top layers of a pretrained backbone for fine-tuning."""


def unfreeze_top_layers(base_model, num_layers):
    """
    Unfreezes the top `num_layers` layers of a pretrained backbone,
    keeping BatchNormalization layers frozen to preserve their
    learned statistics during fine-tuning.

    Args:
        base_model: A Keras Model (e.g. the MobileNetV2 backbone)
            whose layers should be selectively unfrozen.
        num_layers: An integer, the number of layers (counted from
            the end of the model) to unfreeze.

    Returns:
        None. The base_model is modified in place.
    """
    base_model.trainable = True

    freeze_until = len(base_model.layers) - num_layers

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, __import__("tensorflow").keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
