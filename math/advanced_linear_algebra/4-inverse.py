#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Args:
        matrix (list): A list of lists representing a square matrix.

    Returns:
        The inverse of the matrix, or None if the matrix is singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    # 1. Type Check: Ensure matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # 2. Check for Empty List
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # 3. Type Check: Ensure all rows are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 4. Squareness and Content Check
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper function to calculate determinant
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

    # 5. Calculate Determinant
    det = get_determinant(matrix)

    # 6. Check for Singularity (Det == 0)
    if det == 0:
        return None

    # 7. Base Case: 1x1 Matrix
    if n == 1:
        return [[1 / det]]

    # 8. Calculate Cofactor Matrix
    cofactor_matrix = []
    for r in range(n):
        row_cofactors = []
        for c in range(n):
            # Create sub-matrix (Minor)
            sub_matrix = [
                row[:c] + row[c + 1:]
                for i, row in enumerate(matrix) if i != r
            ]
            minor_val = get_determinant(sub_matrix)
            sign = (-1) ** (r + c)
            row_cofactors.append(sign * minor_val)
        cofactor_matrix.append(row_cofactors)

    # 9. Calculate Adjugate (Transpose of Cofactor) and Multiply by 1/det
    inverse_matrix = []
    for c in range(n):
        new_row = []
        for r in range(n):
            # Transpose: use cofactor_matrix[r][c]
            # Divide by determinant
            val = cofactor_matrix[r][c] / det
            new_row.append(val)
        inverse_matrix.append(new_row)

    return inverse_matrix
