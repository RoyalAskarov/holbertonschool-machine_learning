#!/usr/bin/env python3
"""Compute a softmax policy and its sampled action gradient."""
import numpy as np


def policy(matrix, weight):
    """
    Compute action probabilities using a softmax policy.

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


def policy_gradient(state, weight):
    """
    Sample an action and compute its log-policy gradient.

    Args:
        state: Current observation of shape (n,) or (1, n).
        weight: Weight matrix of shape (n, a).

    Returns:
        action: Sampled action index.
        gradient: Gradient of log-policy with shape (n, a).
    """
    state = np.asarray(state).reshape(1, -1)
    probabilities = policy(state, weight)

    action = np.random.choice(
        probabilities.shape[1], p=probabilities[0]
    )

    log_gradient = -probabilities
    log_gradient[0, action] += 1

    gradient = np.matmul(state.T, log_gradient)

    return action, gradient
