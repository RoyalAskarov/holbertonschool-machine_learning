#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list): A list of lists representing a square matrix.

    Returns:
        The determinant of the matrix.
    """
    # 1. Check if matrix is a list and not empty (handles '[]' case)
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # 2. Check if all elements inside are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 3. Special Case: [[]] is defined as the 0x0 matrix with det 1
    if matrix == [[]]:
        return 1

    # 4. Check if matrix is square
    # Now we can safely check dimensions since we passed the list checks
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base Case: 1x1 Matrix
    if n == 1:
        return matrix[0][0]

    # Base Case: 2x2 Matrix
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # Recursive Step (Laplace Expansion)
    det = 0
    for col in range(n):
        # Create minor: remove row 0 and current column
        sub_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub_matrix)

    return det
