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
    """
    # 1. Type Check: Ensure matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Validity Check:
    # - Must be 2-dimensional
    # - Must be square (rows == cols)
    # - Must NOT be empty (size 0) -> This fixes the error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    # 3. Calculate Eigenvalues
    try:
        w = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # 4. Handle Complex Eigenvalues
    # If eigenvalues are complex (and not just real numbers stored as complex),
    # the matrix doesn't fit the standard real definiteness categories.
    if np.iscomplexobj(w) and not np.all(np.isreal(w)):
        return None

    # Convert to real numbers for comparison
    w = np.real(w)

    # 5. Determine Definiteness
    # strictly positive
    if np.all(w > 0):
        return "Positive definite"

    # positive or zero
    if np.all(w >= 0):
        return "Positive semi-definite"

    # strictly negative
    if np.all(w < 0):
        return "Negative definite"

    # negative or zero
    if np.all(w <= 0):
        return "Negative semi-definite"

    # Mixed signs
    return "Indefinite"
