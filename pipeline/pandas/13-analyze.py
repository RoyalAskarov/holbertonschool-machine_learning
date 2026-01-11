#!/usr/bin/env python3
"""
Module that contains a function to analyze a pd.DataFrame
"""


def analyze(df):
    """
    Calculates the mean of all columns, grouped by the Timestamp index
    """
    # Group by level 0 (Timestamp) and calculate the average for each point
    return df.groupby(level=0).mean()
