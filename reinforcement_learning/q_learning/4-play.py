#!/usr/bin/env python3
""" Defines `play`. """
import numpy as np


def play(env, Q_table, max_steps=100):
    """
    Plays an episode of Frozen Lake using a trained agent.

    env: The FrozenLakeEnv instance.
    Q_table: A `numpy.ndarray` representing the Q-table.
    max_steps: The maximum number of steps in the episode.

    Returns: The total rewards for the episode.
    """
    state, _ = env.reset()

    for _step in range(max_steps):
        print(env.render(), end='')

        action = np.argmax(Q_table[state])

        state, reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            break

    print(env.render(), end='')

    return reward
