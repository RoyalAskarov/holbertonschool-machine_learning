#!/usr/bin/env python3
"""
This module contains a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Args:
        array: the np.ndarray from which you should create the pd.DataFrame

    Returns:
        The newly created pd.DataFrame with alphabetical column labels
    """
    # Get the number of columns in the numpy array
    num_cols = array.shape[1]

    # Generate a list of uppercase letters A, B, C... based on column count
    # chr(65) is 'A', chr(66) is 'B', and so on.
    col_names = [chr(i) for i in range(65, 65 + num_cols)]

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=col_names)
