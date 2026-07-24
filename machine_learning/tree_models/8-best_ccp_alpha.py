#!/usr/bin/env python3
"""Select the best ccp_alpha based on test accuracy and generalization"""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best ccp_alpha among trained pruned trees"""
    best_index = None
    best_key = None

    for i in range(len(clfs)):
        gap = train_scores[i] - test_scores[i]
        key = (test_scores[i], -gap, ccp_alphas[i])

        if best_key is None or key > best_key:
            best_key = key
            best_index = i

        best_alpha = ccp_alphas[best_index]
        best_clf = clfs[best_index]

    return best_alpha, best_clf
