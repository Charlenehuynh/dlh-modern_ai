#!/usr/bin/env python3
"""function create a linear regression model using Scikit-learn"""

from sklearn import linear_model


def Linear_Regression():
    """Creates a Linear Regression model using Scikit-learn"""
    model = linear_model.LinearRegression()
    return model
