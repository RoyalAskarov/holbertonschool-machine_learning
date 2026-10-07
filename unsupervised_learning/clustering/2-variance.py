#!/usr/bin/env python3
"""Calculate total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """
    Calculate the sum of squared distances to the nearest centroids.

    Returns:
        var: Total intra-cluster variance.
        None on invalid input.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if X.size == 0 or C.shape[0] == 0:
        return None

    distances_squared = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    var = np.sum(np.min(distances_squared, axis=1))

    return var