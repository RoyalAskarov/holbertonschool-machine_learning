#!/usr/bin/env python3
"""This module contains a function that plots a scatter plot of data"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plots men's height vs weight as a magenta scatter plot"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the data as magenta points
    plt.scatter(x, y, color='magenta')

    # Add labels and title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()