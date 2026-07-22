#!/usr/bin/env python3

"""evaluates K-Means clustering quality using silhouette scores"""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Returns: tuple
    """
    cluster_numbers = []
    inertias = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)

        cluster_numbers.append(k)
        inertias.append(model.inertia_)

        score = metrics.silhouette_score(X, model.labels_)
        silhouette_scores.append(score)

    return cluster_numbers, inertias, silhouette_scores
