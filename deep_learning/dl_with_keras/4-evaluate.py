#!/usr/bin/env python3
"""Assess a trained Keras model's performance"""


def evaluate_model(model, X, Y, verbose=0):
    loss, accurary = model.evaluate(X, Y, verbose=verbose)
    return loss, accurary
