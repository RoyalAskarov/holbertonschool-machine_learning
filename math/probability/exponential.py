#!/usr/bin/env python3
"""
Exponential distribution class
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Exponential distribution.

        Args:
            data (list): A list of data points to estimate the distribution.
            lambtha (float): The expected number of occurrences.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate lambtha from data
            # For Exponential distribution, lambtha is the inverse of the mean (1 / mean)
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)
