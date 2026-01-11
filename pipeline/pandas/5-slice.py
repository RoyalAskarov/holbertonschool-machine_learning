#!/usr/bin/env python3
"""
Module that contains a function to slice a pd.DataFrame
"""


def slice(df):
    """
    Slices a DataFrame to specific columns and every 60th row
    """
    # Note: Using 'Volume_(BTC)' to match the actual CSV header
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df.loc[::60, columns]
