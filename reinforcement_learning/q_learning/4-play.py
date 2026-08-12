#!/usr/bin/env python3
"""Defines the `play` function for a trained agent on FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Plays an episode of Frozen Lake using a trained agent exploiting Q."""
    rendered_outputs = []

    # Safely handle env.reset() for both Gym (returns int) and Gymnasium (returns tuple)
    reset_val = env.reset()
    state = reset_val[0] if isinstance(reset_val, tuple) else reset_val

    # Safely handle render
    try:
        rendered_outputs.append(env.render(mode='ansi'))
    except (TypeError, Exception):
        rendered_outputs.append(env.render())

    total_rewards = 0.0

    for _ in range(max_steps):
        action = np.argmax(Q[state])

        step_val = env.step(action)

        # Safely handle env.step() for both Gym (4 items) and Gymnasium (5 items)
        if len(step_val) == 4:
            state, reward, done, _info = step_val
        else:
            state, reward, terminated, truncated, _info = step_val
            done = terminated or truncated

        total_rewards += reward

        try:
            rendered_outputs.append(env.render(mode='ansi'))
        except (TypeError, Exception):
            rendered_outputs.append(env.render())

        if done:
            break

    return total_rewards, rendered_outputs
