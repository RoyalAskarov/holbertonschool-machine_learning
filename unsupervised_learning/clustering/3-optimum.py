#!/usr/bin/env python3
"""Test cluster sizes using total intra-cluster variance."""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Run K-means for each cluster size from kmin through kmax.

    If kmax is None, use the number of data points.

    Returns:
        results: List of (C, clss) outputs for each cluster size.
        d_vars: Variance reductions relative to kmin.
        (None, None) on invalid input or failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if X.size == 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None
    if kmax > X.shape[0]:
        return None, None

    results = []
    d_vars = []
    initial_variance = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        var = variance(X, C)
        if var is None:
            return None, None

        if initial_variance is None:
            initial_variance = var

        results.append((C, clss))
        d_vars.append(initial_variance - var)

    return results, d_vars
