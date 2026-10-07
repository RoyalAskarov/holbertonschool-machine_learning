#!/usr/bin/env python3
"""Compute action probabilities using a softmax policy."""
import numpy as np


def policy(matrix, weight):
    """
    Compute the policy for a state matrix using its weights.

    Args:
        matrix: State matrix of shape (m, n).
        weight: Weight matrix of shape (n, a).

    Returns:
        Action probabilities of shape (m, a).
    """
    scores = np.matmul(matrix, weight)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)

    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
