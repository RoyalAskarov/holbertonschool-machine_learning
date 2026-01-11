#!/usr/bin/env python3
"""
Module that contains a function to concatenate and restructure DataFrames
"""
import pandas as pd

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """
    jasdiofaodfiopa
    :param df:
    :return:
    """
    # Drop Timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # Compute descriptive statistics
    stats = df.describe()

    return stats
