#!/usr/bin/env python3
"""Create an image classification pipeline from a pre-trained model."""
import transformers


def image_classifier(model):
    """
    Creates a Hugging Face image classification pipeline.

    Args:
        model (str): name of the pre-trained model to use

    Returns:
        classifier: a Hugging Face pipeline object
    """
    classifier = transformers.pipeline("image-classification", model=model)
    return classifier
