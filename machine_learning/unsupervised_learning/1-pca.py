#!/usr/bin/env python3
"""Function perform PCA on tabular data using Scikit-learn"""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
        Return:
        numpy.ndarray: Data transformed into principal component space.
    """
    pca = decomposition.PCA(n_components=n_components,
                            random_state=random_state)
    X_transformed = pca.fit_transform(X)

    return X_transformed, pca
