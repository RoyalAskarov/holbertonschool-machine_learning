#!/usr/bin/env python3
"""
Poisson distribution class
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution.

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

            # Calculate lambtha from data (mean of the data)
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        e = 2.7182818285
        pmf_val = (e ** -self.lambtha) * (self.lambtha ** k) / factorial_k

        return pmf_val

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf_val = 0
        # CDF is the sum of PMF for all values from 0 to k
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
