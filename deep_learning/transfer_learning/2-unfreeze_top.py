#!/usr/bin/env python3
"""
This Module unfreezes the last N layers of the base model.
"""


def unfreeze_top_layers(model, n_layers):
    """
    Unfreeze the last n_layers of the base model inside
    a transfer learning pipeline, and leaves the rest frozen.

    The function should:
    - Assume the base model is the first layer of the input model.
    - Unfreeze the last n_layers of the base model.
    - Leave earlier layers frozen.

    Args:
        model: A full Keras Model with a base model as its first layer.
        n_layers: Integer specifying how many of the last layers in
                  the base model should be unfrozen (set as trainable).

    Returns:
        None
    """
    base_model = model.layers[0]

    if n_layers <= 0 or n_layers > len(base_model.layers):
        raise ValueError(
            "n must be a positive integer and "
            "n must be less or equal to "
            "the number of layers in the base model"
        )

    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True
