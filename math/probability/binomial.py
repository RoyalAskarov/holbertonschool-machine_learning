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
            p_estimate = 1 - (variance / mean)

            # Estimate n (n = Mean / p)
            n_estimate = mean / p_estimate

            # Round n to the nearest integer
            self.n = int(round(n_estimate))

            # Recalculate p using the rounded n
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        # Calculate factorial n!
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i

        # Calculate factorial k!
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        # Calculate factorial (n-k)!
        nk_factorial = 1
        for i in range(1, (self.n - k) + 1):
            nk_factorial *= i

        # Calculate binomial coefficient (n choose k)
        combination = n_factorial / (k_factorial * nk_factorial)

        # Calculate PMF
        pmf_val = combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_val

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of "successes".

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
