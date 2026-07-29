#!/usr/bin/env python3
"""create a Support Vector Machine (SVM) classifier"""

from sklearn import svm


def get_SVM_model(name, random_state):
    """An untrained instance of SVC"""
    model = svm.SVC(kernel=name, random_state=random_state)
    return model
