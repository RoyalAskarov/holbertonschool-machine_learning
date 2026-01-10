#!/usr/bin/env python3
"""
This module contains a function that renames and transforms a pd.DataFrame
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts values to datetime,
    and returns only the Datetime and Close columns.

    Args:
        df: the pd.DataFrame to be transformed

    Returns:
        The modified pd.DataFrame
    """
    # 1. Convert the timestamp values to datetime values
    # unit='s' is used because Unix timestamps are in seconds
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # 2. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # 3. Filter to return only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
