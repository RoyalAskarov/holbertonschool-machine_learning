#!/usr/bin/env python3
"""Select the number of GMM clusters using BIC."""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """
    Find the cluster count with the lowest Bayesian Information Criterion.

    Returns:
        best_k: Best number of clusters.
        best_result: Tuple containing the best model's (pi, m, S).
        log_likelihoods: Log likelihood for each cluster count.
        bic_values: BIC for each cluster count.
        (None, None, None, None) on invalid input or failure.
    """
    failure = (None, None, None, None)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure
    if X.size == 0 or not np.all(np.isfinite(X)):
        return failure

    n, d = X.shape

    if type(kmin) is not int or kmin <= 0 or kmin > n:
        return failure

    if kmax is None:
        kmax = n

    if type(kmax) is not int or kmax < kmin or kmax > n:
        return failure
    if type(iterations) is not int or iterations <= 0:
        return failure
    if not isinstance(tol, float) or not np.isfinite(tol) or tol < 0:
        return failure
    if not isinstance(verbose, bool):
        return failure

    size = kmax - kmin + 1
    log_likelihoods = np.zeros(size)
    bic_values = np.zeros(size)
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if pi is None or m is None or S is None or g is None:
            return failure
        if log_likelihood is None or not np.isfinite(log_likelihood):
            return failure

        parameters = (k - 1) + k * d + k * d * (d + 1) // 2
        index = k - kmin

        log_likelihoods[index] = log_likelihood
        bic_values[index] = parameters * np.log(n) - 2 * log_likelihood
        results.append((pi, m, S))

    best_index = int(np.argmin(bic_values))
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log_likelihoods, bic_values
