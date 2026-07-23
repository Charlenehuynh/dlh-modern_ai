#!/usr/bin/env python3
"""train a tree-based classifier using Scikit-learn"""


def train_tree(clf, X, y):
    """
    clf: A Scikit-learn classifier instance
    X: Input features
    y: Target labels
    """
    clf.fit(X, y)
