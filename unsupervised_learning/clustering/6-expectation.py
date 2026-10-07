#!/usr/bin/env python3
"""Calculate the expectation step for a Gaussian Mixture Model."""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate posterior probabilities and total log likelihood.

    Returns:
        g: Posterior probabilities of shape (k, n).
        l: Total log likelihood.
        (None, None) on invalid input or failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if X.size == 0:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if k == 0 or not np.all(np.isfinite(pi)):
        return None, None
    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None
    if not isinstance(m, np.ndarray) or m.shape != (k, d):
        return None, None
    if not isinstance(S, np.ndarray) or S.shape != (k, d, d):
        return None, None

    weighted = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        weighted[i] = pi[i] * P

    total = np.sum(weighted, axis=0)
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        return None, None

    g = weighted / total
    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
