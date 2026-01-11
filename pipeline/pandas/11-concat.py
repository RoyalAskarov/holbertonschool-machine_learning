#!/usr/bin/env python3
"""
Module that contains a function to concatenate two pd.DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes with specific indexing and filtering
    """
    # 1. Index both dataframes on their Timestamp columns
    df1 = index(df1)
    df2 = index(df2)

    # 2. Filter df2 to include timestamps up to 1417411920
    df2_filtered = df2.loc[:1417411920]

    # 3. Concatenate df2 (bitstamp) to the top of df1 (coinbase) with keys
    return pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
