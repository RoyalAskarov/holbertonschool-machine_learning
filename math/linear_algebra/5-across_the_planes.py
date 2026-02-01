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
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return mat1 + mat2
