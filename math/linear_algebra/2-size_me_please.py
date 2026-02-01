#!/usr/bin/env python3
"""
Module to calculate the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix
    Args:
        matrix: A multi-dimensional list
    Returns:
        A list of integers representing the dimensions
    """
    shape = []
    # Continue diving into the first element until it's no longer a list
    while isinstance(matrix, list):
        shape.append(len(matrix))
        # Safely move to the next inner dimension if it exists
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape
