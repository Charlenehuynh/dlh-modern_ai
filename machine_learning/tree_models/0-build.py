#!/usr/bin/env python3
"""Create a decision tree classifier"""

from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """ Create a decision tree classifier"""
    n_tree = tree.DecisionTreeClassifier(
        min_samples_leaf=min_samples_split,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return n_tree
