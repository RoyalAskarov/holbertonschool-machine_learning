#!/usr/bin/python3
"""
Module to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    Args:
        poly: list of coefficients representing a polynomial
        C: integer representing the integration constant
    Returns:
        A new list of coefficients representing the integral
    """
    # Validate poly is a list and C is an integer
    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None

    # Validate all elements in poly are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # The integral starts with the constant C at index 0
    integral = [C]

    for i in range(len(poly)):
        # Calculate the new coefficient: coefficient / (power + 1)
        new_coeff = poly[i] / (i + 1)

        # Requirement: represent as integer if it's a whole number
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)

        integral.append(new_coeff)

    # Requirement: the returned list should be as small as possible
    # This means removing trailing zeros if the original poly ended in zeros
    # while len(integral) > 1 and integral[-1] == 0:
    #     integral.pop()

    return integral
