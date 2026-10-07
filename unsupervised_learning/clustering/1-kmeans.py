#!/usr/bin/env python3
"""Perform K-means clustering."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on X.

    Returns:
        C: Cluster centroids of shape (k, d).
        clss: Cluster indices of shape (n,).
        (None, None) on invalid input.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if X.shape[1] == 0:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, X.shape[1]))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        previous = C.copy()

        for j in range(k):
            points = X[clss == j]
            if points.shape[0] == 0:
                C[j] = np.random.uniform(low, high, size=(X.shape[1],))
            else:
                C[j] = points.mean(axis=0)

        if np.array_equal(C, previous):
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
