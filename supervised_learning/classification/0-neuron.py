#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Class constructor
        Args:
            nx: the number of input features to the neuron
        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights initialized using a random normal distribution
        # Shape is (1, nx) to match the expected output in the main file
        self.__W = np.random.randn(1, nx)

        # Bias initialized to 0
        self.__b = 0

        # Activated output initialized to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    @A.setter
    def A(self, value):
        """Setter for the activated output (needed for the test script)"""
        self.__A = value
