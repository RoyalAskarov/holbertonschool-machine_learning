#!/usr/bin/env python3
"""
Module that contains a function to flip and switch a pd.DataFrame
"""
import pandas as pd


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it

    Args:
        df: the pd.DataFrame to be transformed

    Returns:
        the transformed pd.DataFrame
    )"""
    # 1. Sort the data in reverse chronological order by its index
    # 2. Transpose the result so rows become columns and vice versa
    return df.sort_index(ascending=False).transpose()
