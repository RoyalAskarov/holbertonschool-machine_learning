#!/usr/bin/env python3
"""
This module contains a function that converts parts of a DataFrame to a numpy ndarray
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns and converts them to a numpy ndarray

    Args:
        df: pd.DataFrame containing columns High and Close

    Returns:
        numpy.ndarray containing the last 10 rows of the specified columns
    """
    # 1. Select the required columns: High and Close
    # 2. Use .tail(10) to get only the last 10 rows
    # 3. Use .to_numpy() to convert the selection into a numpy array
    return df[['High', 'Close']].tail(10).to_numpy()
