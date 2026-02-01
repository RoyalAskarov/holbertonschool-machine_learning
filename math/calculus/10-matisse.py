#!/usr/bin/python3
"""
Module to calculate the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    Args:
        poly: list of coefficients representing a polynomial
    Returns:
        A new list of coefficients representing the derivative
    """
    # Validate that poly is a non-empty list of numbers
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # If the polynomial is a constant (e.g., [5]), the derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative: multiply coefficient by its index (power)
    # The first element (index 0) is dropped as its derivative is 0
    derivative = [poly[i] * i for i in range(1, len(poly))]

    return derivative
