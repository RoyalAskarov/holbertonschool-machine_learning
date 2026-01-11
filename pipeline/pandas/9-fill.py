#!/usr/bin/env python3
"""
Module that contains a function to fill missing values in a pd.DataFrame
"""


def fill(df):
    """
    Fills missing values in a DataFrame according to specific logic
    """
    # 1. Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # 2. Fill missing Close values with the previous row's value
    df['Close'] = df['Close'].ffill()

    # 3. Fill High, Low, and Open with the Close value in the same row
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Set missing values in Volume columns to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
