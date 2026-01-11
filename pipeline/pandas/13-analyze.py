#!/usr/bin/env python3
"""
Module that contains a function to visualize a pd.DataFrame
"""
import matplotlib.pyplot as plt
import pandas as pd


def visualize(df):
    """
    Visualizes the price data from 2017 and beyond at daily intervals
    """
    # 1. Filter the data to only include 2017 and beyond
    # 1483228800 is the Unix timestamp for 2017-01-01
    df = df.loc[df.index >= 1483228800]

    # 2. Convert the index to datetime objects for readable axis labels
    df.index = pd.to_datetime(df.index, unit='s')

    # 3. Resample the data to daily (D) intervals, taking the mean
    df_daily = df.resample('D').mean()

    # 4. Plot High, Low, Open, and Close
    # We separate the columns to keep the line length under 79 chars
    cols = ['High', 'Low', 'Open', 'Close']
    df_daily[cols].plot()

    # 5. Set the required titles and labels
    plt.title('Bitcoin Price Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')

    # Display the plot
    plt.show()
