#!/usr/bin/env python3
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has a trained agent play an episode.

    Args:
        env: FrozenLakeEnv instance
        Q: Q-table
        max_steps: Maximum number of steps

    Returns:
        total_rewards: Total rewards for the episode
        rendered_outputs: List of rendered board states
    """
    rendered_outputs = []
    total_rewards = 0

    # Display the initial state
    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        # Always exploit: choose the action with the highest Q-value
        action = np.argmax(Q[env.unwrapped.s])

        # Take the action
        _, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward

        # Render the new state
        rendered_outputs.append(env.render())

        # Stop if the episode has ended
        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
