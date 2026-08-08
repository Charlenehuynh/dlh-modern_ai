#!/usr/bin/env python3
"""Assess a trained Keras model's performance"""


def evaluate_model(model, X, Y, verbose=0):
    """
    loss: The calculated loss on the provided data.
    accuracy: The accuracy of the model on the provided data.
    """
    loss, accurary = model.evaluate(X, Y, verbose=verbose)
    return loss, accurary
