#!/usr/bin/env python3
"""function standardizes tabular data using Scikit-learn."""

from sklearn import preprocessing


def Standardize(X):
    """Return standardized version of the input data"""
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
