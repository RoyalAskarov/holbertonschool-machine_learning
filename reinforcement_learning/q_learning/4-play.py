#!/usr/bin/env python3
"""Defines the `play` function for a trained agent on FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Plays an episode of Frozen Lake using a trained agent exploiting Q.

    Args:
        env: The FrozenLakeEnv instance.
        Q: A numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: Total rewards for the episode.
        rendered_outputs: List of rendered outputs representing board states.
    """
    rendered_outputs = []

    # Reset environment and capture initial state/render
    state, _ = env.reset()
    rendered_outputs.append(env.render())

    total_rewards = 0.0

    for _ in range(max_steps):
        # Always exploit the Q-table
        action = np.argmax(Q[state])

        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward

        # Record board state after taking the step
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
