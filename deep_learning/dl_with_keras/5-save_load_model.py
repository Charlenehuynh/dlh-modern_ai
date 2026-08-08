#!/usr/bin/env python3
"""functions to save and reload a Keras model"""

from tensorflow import keras


def save_model(model, filepath):
    """function to save model"""
    model.save(filepath)


def load_model(file_path):
    """function to load model"""
    load = keras.models.load_model(file_path)
    return load
