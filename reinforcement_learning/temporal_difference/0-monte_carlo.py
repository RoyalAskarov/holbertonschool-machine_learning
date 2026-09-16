#!/usr/bin/env python3
"""Estimate state values using first-visit Monte Carlo."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Update and return state values using first-visit Monte Carlo.

    Args:
        env: Gymnasium environment.
        V: NumPy array containing state value estimates.
        policy: Function mapping a state to an action.
        episodes: Number of training episodes.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        The updated value array V.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        trajectory = []
        first_visits = {}

        for step in range(max_steps):
            first_visits.setdefault(state, step)
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        total_return = 0.0
        for step in range(len(trajectory) - 1, -1, -1):
            state, reward = trajectory[step]
            total_return = reward + gamma * total_return

            if first_visits[state] == step:
                V[state] += alpha * (total_return - V[state])

    return V
