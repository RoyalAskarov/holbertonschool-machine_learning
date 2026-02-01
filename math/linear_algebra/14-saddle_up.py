#!/usr/bin/env python3
"""
Module to perform matrix multiplication using numpy
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication
    Args:
        mat1: a numpy.ndarray
        mat2: a numpy.ndarray
    Returns:
        a new numpy.ndarray representing the product of mat1 and mat2
    """
    return mat1 @ mat2
