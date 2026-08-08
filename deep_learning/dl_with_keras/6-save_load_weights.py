#!/usr/bin/env python3
"""2 functions to save and reload the weights of a trained Keras mode"""


def save_model_weights(model, filepath):
    """
    model: A trained Keras model whose weights need to be saved.
    filepath: A string representing the file path
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """
        model: A Keras model whereweights will be loaded.
        filepath: A string representing the file path
    """
    model.load_weights(filepath)
