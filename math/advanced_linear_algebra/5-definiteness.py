#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): A numpy array of shape (n, n).

    Returns:
        str: The definiteness category ('Positive definite',
             'Positive semi-definite', 'Negative semi-definite',
             'Negative definite', 'Indefinite').
        None: If the matrix is not valid or does not fit a category.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    # 1. Type Check: Ensure matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Validity Check:
    # - Must be 2-dimensional (handles empty array np.array([]) which is 1D)
    # - Must be square (rows == cols)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # 3. Calculate Eigenvalues
    # We calculate the eigenvalues to check the definiteness properties.
    try:
        w = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # 4. Handle Complex Eigenvalues
    # Definiteness categories generally apply to matrices with real eigenvalues
    # (e.g., symmetric matrices). If eigenvalues are complex, return None.
    if np.iscomplexobj(w) and not np.all(np.isreal(w)):
        return None

    # Ensure we are working with the real parts (floating point precision)
    w = np.real(w)

    # 5. Determine Definiteness based on signs of eigenvalues
    # Check strictly positive first
    if np.all(w > 0):
        return "Positive definite"

    # Check non-negative
    if np.all(w >= 0):
        return "Positive semi-definite"

    # Check strictly negative
    if np.all(w < 0):
        return "Negative definite"

    # Check non-positive
    if np.all(w <= 0):
        return "Negative semi-definite"

    # If eigenvalues have mixed signs (some positive, some negative)
    # and none of the above conditions were met, it is Indefinite.
    # (e.g., eigenvalues like [3, -1])
    return "Indefinite"
