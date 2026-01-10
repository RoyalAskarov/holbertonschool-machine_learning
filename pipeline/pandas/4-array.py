#!/usr/bin/env python3
"""
Module to convert parts of a DataFrame to a numpy ndarray
"""


def array(df):
    """
    Selects last 10 rows of High and Close columns and
    converts them to a numpy ndarray
    """
    return df[['High', 'Close']].tail(10).to_numpy()