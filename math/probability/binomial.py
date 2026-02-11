#!/usr/bin/env python3
"""
Binomial distribution class
"""


class Binomial:
    """
    Represents a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.

        Args:
            data (list): A list of data points to estimate the distribution.
            n (int): The number of Bernoulli trials.
            p (float): The probability of a "success".
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate sample mean
            mean = float(sum(data) / len(data))

            # Calculate sample variance
            sum_squared_diff = 0
            for x in data:
                sum_squared_diff += (x - mean) ** 2
            variance = float(sum_squared_diff / len(data))

            # Estimate p (p = 1 - (variance / mean))
            # Derived from: Mean = np, Var = np(1-p) -> Var/Mean = 1-p
            p_estimate = 1 - (variance / mean)

            # Estimate n (n = Mean / p)
            n_estimate = mean / p_estimate

            # Round n to the nearest integer
            self.n = int(round(n_estimate))

            # Recalculate p using the rounded n to ensure Mean = np holds
            self.p = float(mean / self.n)
