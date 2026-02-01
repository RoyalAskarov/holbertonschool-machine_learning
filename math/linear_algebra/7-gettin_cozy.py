#!/usr/bin/env python3
"""
Module to concatenate two 2D matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis
    Args:
        mat1: First 2D list of ints/floats
        mat2: Second 2D list of ints/floats
        axis: The axis along which to concatenate (0 or 1)
    Returns:
        A new 2D matrix, or None if they cannot be concatenated
    """
    if axis == 0:
        # Check if they have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Return a new list combining rows from both
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Check if they have the same number of rows
        if len(mat1) != len(mat2):
            return None
        # Return a new list joining rows element-wise
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
