#!/usr/bin/env python3
"""
Monte Carlo evaluation module.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value estimate.

    Args:
        env: The environment instance.
        V: A numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns the next action.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V, the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        # Calculate returns and update V (Every-visit Monte Carlo)
        G = 0
        for s, r in reversed(episode):
            G = gamma * G + r
            V[s] = V[s] + alpha * (G - V[s])

    return V