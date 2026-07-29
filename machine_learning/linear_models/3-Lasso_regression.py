#!/usr/bin/env python3
"""model creates and returns a Lasso Regression"""


def lasso_regression(random_state):
    """Creates and returns an untrained Lasso regression model instance."""
    model = linear_model.Lasso(random_state=random_state)
    return model
