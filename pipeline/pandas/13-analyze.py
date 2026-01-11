#!/usr/bin/env python3
"""
gfsdesr
"""


def analyze(df):
    """
    asdfas
    :param df:
    :return:
    """
    # Drop Timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # Compute descriptive statistics
    return df.describe()
