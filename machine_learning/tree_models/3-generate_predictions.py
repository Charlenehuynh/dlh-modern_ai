#!/usr/bin/env python3
"""function generate_predictions(clf, X)"""


def generate_predictions(clf, X):
    """return array contain predicted class labels"""
    prediction = clf.predict(X)
    return prediction
