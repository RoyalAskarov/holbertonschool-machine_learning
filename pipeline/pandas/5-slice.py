#!/usr/bin/env python3
"""
Module that contains a function to slice a pd.DataFrame
"""


def slice(df):
    """
    Slices a DataFrame to specific columns and every 60th row
    """
    # Select the required columns and step through rows by 60
    columns = ['High', 'Low', 'Close', 'Volume_BTC']
    return df.loc[::60, columns]
