#!/usr/bin/env python3
"""
Calculates the marginal probability of obtaining data
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining x and n
    """
    # Order of exceptions is strictly enforced per requirements
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        # x is used as a placeholder in the prompt's error message requirement
        for i, val in enumerate(P):
            if val < 0 or val > 1:
                raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        for i, val in enumerate(Pr):
            if val < 0 or val > 1:
                raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # 1. Calculate Likelihood using Binomial PMF
    # Formula: (n! / (x!(n-x)!)) * (p^x) * (1-p)^(n-x)
    fact = np.math.factorial
    n_cr = fact(n) / (fact(x) * fact(n - x))
    likelihood = n_cr * (P ** x) * ((1 - P) ** (n - x))

    # 2. Calculate Intersections: Likelihood * Prior
    intersection = likelihood * Pr

    # 3. Calculate Marginal Probability: Sum of all intersections
    return np.sum(intersection)
