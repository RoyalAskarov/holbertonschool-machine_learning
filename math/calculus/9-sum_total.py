#!/usr/bin/python3
"""
Module to calculate the summation of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n using Faulhaber's formula.
    Args:
        n: The stopping condition (integer)
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    # Check if n is a valid number (should be an integer and at least 1)
    if not isinstance(n, int) or n < 1:
        return None

    # Calculate using the formula: [n(n+1)(2n+1)] / 6
    # We use integer division // to ensure the result is an integer
    result = (n * (n + 1) * (2 * n + 1)) // 6

    return result
