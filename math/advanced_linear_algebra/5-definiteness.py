#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    fdggdsgf
    :param matrix:
    :return:
    """
    # Type check
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Must be 2D, square, non-empty
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    # Matrix must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        w = np.linalg.eigvalsh(matrix)  # better for symmetric matrices
    except Exception:
        return None

    if np.all(w > 0):
        return "Positive definite"

    if np.all(w >= 0):
        return "Positive semi-definite"

    if np.all(w < 0):
        return "Negative definite"

    if np.all(w <= 0):
        return "Negative semi-definite"

    return "Indefinite"
