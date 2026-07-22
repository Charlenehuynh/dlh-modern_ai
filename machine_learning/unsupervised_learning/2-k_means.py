#!/usr/bin/env python3

"""
Creates and fits a K-Means clustering model on tabular data
"""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Returns:
    sklearn.cluster.KMeans: K-Means instance fitted on the input data
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans
