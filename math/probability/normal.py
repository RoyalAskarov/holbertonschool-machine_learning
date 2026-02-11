#!/usr/bin/env python3
"""
Normal distribution class
"""


class Normal:
    """
    Represents a normal distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.

        Args:
            data (list): A list of data points to estimate the distribution.
            mean (float): The mean of the distribution.
            stddev (float): The standard deviation of the distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate standard deviation
            sum_squared_diff = 0
            for x in data:
                sum_squared_diff += (x - self.mean) ** 2

            self.stddev = float((sum_squared_diff / len(data)) ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x: The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z: The z-score.

        Returns:
            float: The x-value of z.
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Args:
            x: The x-value.

        Returns:
            float: The PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536

        # Calculate the coefficient term (1 / (σ * √2π))
        denominator = self.stddev * ((2 * pi) ** 0.5)
        coefficient = 1 / denominator

        # Calculate the exponent term (-1/2 * ((x-μ)/σ)^2)
        # Note: (x - mean) / stddev is the z-score
        z = (x - self.mean) / self.stddev
        exponent = -0.5 * (z ** 2)

        # Combine
        pdf_val = coefficient * (e ** exponent)

        return pdf_val