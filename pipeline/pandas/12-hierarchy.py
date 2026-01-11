#!/usr/bin/env python3
"""
Module that contains a function to concatenate and restructure DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes and rearranges the MultiIndex hierarchy
    """
    # 1. Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # 2. Filter both dataframes for the specific range inclusive
    df1_filtered = df1.loc[1417411980:1417417980]
    df2_filtered = df2.loc[1417411980:1417417980]

    # 3. Concatenate with exchange keys
    # Index starts as Level 0: Exchange, Level 1: Timestamp
    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

    # 4. Swap levels so Timestamp is first
    df = df.swaplevel(0, 1)

    # 5. Sort to ensure chronological order
    df = df.sort_index()

    return df
