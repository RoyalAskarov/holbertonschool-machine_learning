#!/usr/bin/env python3
"""Calculate the maximization step for a Gaussian Mixture Model."""
import numpy as np


def maximization(X, g):
    """
    Update cluster priors, means, and covariance matrices.

    Returns:
        pi: Updated priors of shape (k,).
        m: Updated means of shape (k, d).
        S: Updated covariance matrices of shape (k, d, d).
        (None, None, None) on invalid input or failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if X.size == 0 or not np.all(np.isfinite(X)):
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if k == 0 or g.shape[1] != n:
        return None, None, None
    if not np.all(np.isfinite(g)) or np.any(g < 0):
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    counts = np.sum(g, axis=1)
    if np.any(counts <= 0):
        return None, None, None

    pi = counts / n
    m = np.matmul(g, X) / counts[:, np.newaxis]
    S = np.zeros((k, d, d))

    for i in range(k):
        centered = X - m[i]
        weighted = centered * g[i, :, np.newaxis]
        S[i] = np.matmul(weighted.T, centered) / counts[i]

    return pi, m, S
