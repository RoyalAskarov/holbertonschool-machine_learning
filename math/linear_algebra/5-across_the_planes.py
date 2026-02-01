#!/usr/bin/env python3
import numpy as np
"""
dfsjsa
"""


def add_matrices2D(mat1, mat2):
    """
    dsfjsld;a
    :param mat1:
    :param mat2:
    :return:
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1))]
            for i in range(len(mat1))]
