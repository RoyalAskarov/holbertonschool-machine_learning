#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.

    Args:
        matrix (list): A list of lists representing a square matrix.

    Returns:
        The minor matrix of the input matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    # 1. Type Check: Ensure matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # 2. Check for Empty List: '[]' is invalid for this specific task
    # The output shows 'minor(mat5)' where mat5=[] raises "matrix must be a list of lists"
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # 3. Type Check: Ensure all rows are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 4. Squareness and Content Check
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper function to calculate determinant (reused from previous logic)
    def get_determinant(m):
        if len(m) == 1:
            return m[0][0]
        if len(m) == 2:
            return (m[0][0] * m[1][1]) - (m[0][1] * m[1][0])

        det = 0
        for c in range(len(m)):
            sub_m = [r[:c] + r[c + 1:] for r in m[1:]]
            det += ((-1) ** c) * m[0][c] * get_determinant(sub_m)
        return det

    # 5. Base Case: 1x1 Matrix
    # The minor of a 1x1 matrix is implicitly [[1]] in many definitions,
    # and the output shows 'minor([[5]])' -> [[1]]
    if n == 1:
        return [[1]]

    # 6. Calculate Minor Matrix
    minor_matrix = []
    for r in range(n):
        row_minors = []
        for c in range(n):
            # Create sub-matrix by removing row 'r' and column 'c'
            sub_matrix = [row[:c] + row[c + 1:] for i, row in enumerate(matrix) if i != r]

            # Calculate determinant of sub-matrix
            det = get_determinant(sub_matrix)
            row_minors.append(det)
        minor_matrix.append(row_minors)

    return minor_matrix
