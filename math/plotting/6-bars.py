#!/usr/bin/env python3
"""Module to plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots stacked bars of fruit per person with specific colors"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Initialize bottom tracking for stacking
    bottom = np.zeros(3)
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=fruit_types[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend()
    plt.show()
