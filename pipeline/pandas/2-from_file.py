#!/usr/bin/env python3
"""
This module contains a function that loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Args:
        filename: the file to load from
        delimiter: the column separator

    Returns:
        the loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
