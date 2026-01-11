#!/usr/bin/env python3
"""
Module that contains a function to sort a pd.DataFrame
"""


def high(df):
    """
    Sorts a DataFrame by the High price in descending order
    """
    # Sort by the 'High' column and set ascending to False for descending
    return df.sort_values(by='High', ascending=False)
