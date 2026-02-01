#!/usr/bin/env python3
"""
Module to perform element-wise arithmetic on numpy.ndarrays
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    Args:
        mat1: a numpy.ndarray
        mat2: a numpy.ndarray
    Returns:
        a tuple containing the element-wise sum, difference, product,
        and quotient, respectively
    """
    sum_result = mat1 + mat2
    diff_result = mat1 - mat2
    prod_result = mat1 * mat2
    quot_result = mat1 / mat2

    return (sum_result, diff_result, prod_result, quot_result)
