#!/usr/bin/env python3
"""
Module to perform matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    Args:
        mat1: First 2D list of ints/floats
        mat2: Second 2D list of ints/floats
    """
    # Validation: Columns in mat1 must equal rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    cols_mat2 = len(mat2[0])

    result=[[0 for _ in range(cols_mat2)]
            for _ in range(rows_mat1)]

    for i in range(rows_mat1):
        for j in range(cols_mat2):
            for k in range(cols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
