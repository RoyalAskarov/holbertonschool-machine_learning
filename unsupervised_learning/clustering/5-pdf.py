#!/usr/bin/env python3
"""Calculate the PDF of a multivariate Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """
    Evaluate the Gaussian probability density at each data point.

    Returns:
        P: PDF values of shape (n,), with a minimum of 1e-300.
        None on invalid input or failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    d = X.shape[1]
    if d == 0:
        return None
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None
    if not np.allclose(S, S.T):
        return None

    try:
        L = np.linalg.cholesky(S)
        centered = X - m
        transformed = np.linalg.solve(L, centered.T)
        squared_distances = np.sum(transformed ** 2, axis=0)
        _, log_det = np.linalg.slogdet(S)
    except (np.linalg.LinAlgError, TypeError, ValueError):
        return None

    log_P = -0.5 * (
        d * np.log(2 * np.pi) + log_det + squared_distances
    )
    P = np.exp(log_P)
    return np.maximum(P, 1e-300)

