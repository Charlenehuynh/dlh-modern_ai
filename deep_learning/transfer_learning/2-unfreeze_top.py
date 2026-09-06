#!/usr/bin/env python3
"""
This Module unfreezes the last N layers of the base model.
"""


def unfreeze_top_layers(model, n_layers):
    """
    Unfreeze the last n_layers of the base model, and leaves
    the rest frozen.

    Args:
        model: The base model (e.g. MobileNetV2 backbone) whose
               layers should be selectively unfrozen.
        n_layers: Integer specifying how many of the last layers in
                  the base model should be unfrozen (set as trainable).

    Returns:
        None
    """
    if n_layers <= 0 or n_layers > len(model.layers):
        raise ValueError(
            "n must be a positive integer and "
            "n must be less or equal to "
            "the number of layers in the base model"
        )

    for layer in model.layers[-n_layers:]:
        layer.trainable = True
