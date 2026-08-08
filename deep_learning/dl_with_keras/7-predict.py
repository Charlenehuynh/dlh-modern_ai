#!/usr/bin/env python3
"""make predictions on a given dataset using a trained Keras model"""

import tensorflow as tf


def predict(model, X, verbose=0):
    """return a list of predicted class labels for the input data"""
    predictions = model.predict(X, verbose=verbose)
    labels = tf.argmax(predictions, axis=1)
    return labels
