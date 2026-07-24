#!/usr/bin/env python3
"""function retrieves the cost-complexity pruning path"""


def get_pruning_path(clf, X, y):
    """
    clf: A DecisionTreeClassifier instance
    X: Input features
    y: Target labels
    """
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    return ccp_alphas, impurities
