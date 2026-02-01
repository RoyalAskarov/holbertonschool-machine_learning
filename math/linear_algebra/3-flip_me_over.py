#!/usr/bin/env python3
"""
Module to transpose a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    """
    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]
