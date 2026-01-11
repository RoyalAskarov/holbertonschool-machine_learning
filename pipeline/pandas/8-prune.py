#!/usr/bin/env python3
"""
Module that contains a function to prune a pd.DataFrame
"""


def prune(df):
    """
    Removes entries where the Close column has NaN values
    """
    # dropna with subset ensures we only filter based on the Close column
    return df.dropna(subset=['Close'])
