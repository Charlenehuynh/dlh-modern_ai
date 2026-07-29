#!/usr/bin/env python3
"""
Creates and returns an untrained Ridge regression model instance.
"""

from sklearn import linear_model


def ridge_regression(random_state):
    """Module that initializes a Ridge Regression model."""
    model = linear_model.Ridge(random_state=random_state)
    return model
