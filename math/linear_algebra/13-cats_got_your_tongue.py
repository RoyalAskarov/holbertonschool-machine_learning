#!/usr/bin/env python3
"""
Module to concatenate two numpy.ndarrays along a specific axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Args:
        mat1: a numpy.ndarray
        mat2: a numpy.ndarray
        axis: the axis along which the matrices should be concatenated
    Returns:
        a new numpy.ndarray that is the concatenation of mat1 and mat2
    """
    return np.concatenate((mat1, mat2), axis=axis)