#!/usr/bin/env python3
"""
dgsfdfgsdf
"""
import matplotlib.pyplot as plt
import pandas as pd


def visualize(df):
    # 1. Remove Weighted_Price
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # 2. Rename Timestamp to Date
    df = df.rename(columns={'Timestamp': 'Date'})

    # 3. Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # 4. Index the dataframe on Date
    df = df.set_index('Date')

    # 5. Fill missing Close with previous value
    df['Close'] = df['Close'].fillna(method='ffill')

    # 6. Fill missing High, Low, Open with Close
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    # 7. Fill missing Volume columns with 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    # 8. Filter from 2017 onward
    df = df[df.index.year >= 2017]

    # 9. Group by day with required aggregations
    daily = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # 10. Plot
    daily[['Open', 'Close', 'High', 'Low']].plot()
    plt.show()

    # 11. Return transformed dataframe before plotting
    return daily
