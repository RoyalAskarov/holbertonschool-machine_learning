#!/usr/bin/env python3
"""Perform expectation maximization for a Gaussian Mixture Model."""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Fit a Gaussian Mixture Model using expectation maximization.

    Returns:
        pi: Cluster priors of shape (k,).
        m: Cluster means of shape (k, d).
        S: Covariance matrices of shape (k, d, d).
        g: Posterior probabilities of shape (k, n).
        log_likelihood: Total log likelihood.
        (None, None, None, None, None) on invalid input or failure.
    """
    failure = (None, None, None, None, None)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return failure
    if X.size == 0 or not np.all(np.isfinite(X)):
        return failure
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return failure
    if type(iterations) is not int or iterations <= 0:
        return failure
    if not isinstance(tol, float) or not np.isfinite(tol) or tol < 0:
        return failure
    if not isinstance(verbose, bool):
        return failure

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return failure

    previous_log_likelihood = None

    for i in range(iterations + 1):
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return failure
        if not np.isfinite(log_likelihood):
            return failure

        converged = (
            previous_log_likelihood is not None
            and abs(log_likelihood - previous_log_likelihood) <= tol
        )
        finished = converged or i == iterations

        if verbose and (i % 10 == 0 or finished):
            print(
                "Log Likelihood after {} iterations: {:.5f}".format(
                    i, log_likelihood
                )
            )

        if finished:
            return pi, m, S, g, log_likelihood

        previous_log_likelihood = log_likelihood
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return failure
