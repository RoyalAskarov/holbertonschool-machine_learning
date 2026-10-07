#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize GMM priors, means, and covariance matrices.

    Returns:
        pi: Equal cluster priors of shape (k,).
        m: K-means centroids of shape (k, d).
        S: Identity covariance matrices of shape (k, d, d).
        (None, None, None) on invalid input or failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if X.size == 0:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    m, clss = kmeans(X, k)
    if m is None or clss is None:
        return None, None, None

    pi = np.full(k, 1.0 / k)
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
