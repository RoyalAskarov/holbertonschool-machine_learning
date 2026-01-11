#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plots y as a solid red line graph with the x-axis ranging from 0 to 10"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Plot y as a solid red line
    # If x is not specified, matplotlib uses the index of y (0 to 10)
    plt.plot(y, color='red', linestyle='solid')

    # Set the x-axis range from 0 to 10
    plt.xlim(0, 10)

    # Display the plot
    plt.show()
