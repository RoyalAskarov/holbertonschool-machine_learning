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
        det = np.linalg.det(S)
        if det <= 0:
            return None

        inv = np.linalg.inv(S)
        centered = X - m
        exponent = -0.5 * np.sum(
            np.matmul(centered, inv) * centered, axis=1
        )
        denominator = np.sqrt((2 * np.pi) ** d * det)
        P = np.exp(exponent) / denominator
    except (np.linalg.LinAlgError, TypeError, ValueError):
        return None

    return np.maximum(P, 1e-300)
