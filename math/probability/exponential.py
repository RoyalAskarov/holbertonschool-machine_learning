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

            # Calculate lambtha from data (1 / mean)
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Args:
            x: The time period.

        Returns:
            float: The PDF value for x.
        """
        # The domain of an Exponential distribution is x >= 0
        if x < 0:
            return 0

        e = 2.7182818285

        # Formula: f(x) = lambtha * e^(-lambtha * x)
        pdf_val = self.lambtha * (e ** (-self.lambtha * x))

        return pdf_val
