#!/usr/bin/env python3

import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table.

    Args:
        env: FrozenLakeEnv instance
        Q: Q-table
        max_steps: Maximum number of steps.

    Returns:
        The total rewards and a list of rendered outputs.
    """
    total_rewards = 0
    rendered_outputs = []

    # Render initial state
    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        # Always exploit the Q-table
        state = env.unwrapped.s
        action = np.argmax(Q[state])

        # Take action
        _, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward

        # Render current state
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
