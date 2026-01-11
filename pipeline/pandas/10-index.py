#!/usr/bin/env python3
"""
Module that contains a function to index a pd.DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe
    """
    # Returns a new DataFrame with 'Timestamp' as the index labels
    return df.set_index('Timestamp')
