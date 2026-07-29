#!/usr/bin/env python3
"""create a logistic regression model using Scikit-learn"""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """return An untrained LogisticRegression instance."""
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
