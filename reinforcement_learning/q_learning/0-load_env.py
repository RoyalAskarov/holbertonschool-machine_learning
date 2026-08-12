#!/usr/bin/env python3
"""Module to load Gymnasium's FrozenLake environment."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the FrozenLakeEnv environment from Gymnasium.

    Args:
        desc: list of lists containing a custom map description or None
        map_name: string containing pre-made map name to load or None
        is_slippery: boolean to determine if ice is slippery

    Returns:
        The created Gymnasium FrozenLake environment
    """
    if desc is None and map_name is None:
        map_name = '8x8'

    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
