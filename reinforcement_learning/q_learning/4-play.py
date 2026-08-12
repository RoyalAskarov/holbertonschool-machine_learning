#!/usr/bin/env python3
"""Defines the `play` function for a trained agent on FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Plays an episode of Frozen Lake using a trained agent exploiting Q."""

    # Reset environment safely
    reset_val = env.reset()
    # Handle both Gymnasium (returns tuple) and older Gym (returns int)
    state = reset_val[0] if isinstance(reset_val, tuple) else reset_val

    rendered_outputs = [env.render()]

    total_rewards = 0.0

    for _ in range(max_steps):
        # Always exploit the Q-table
        action = np.argmax(Q[state])

        step_val = env.step(action)

        # Unpack safely for both Gymnasium (5 variables) and Gym (4 variables)
        if len(step_val) == 5:
            state, reward, terminated, truncated, _ = step_val
            done = terminated or truncated
        else:
            state, reward, done, _ = step_val

        total_rewards += reward
        rendered_outputs.append(env.render())

        if done:
            break

    return total_rewards, rendered_outputs
