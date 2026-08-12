#!/usr/bin/env python3
"""Module to load Gymnasium/Gym's FrozenLake environment."""
# Try importing gym, fallback to gymnasium if needed by your local setup
try:
    import gym
except ImportError:
    import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the FrozenLakeEnv environment."""
    if desc is None and map_name is None:
        map_name = '8x8'

    # Safely handle the env creation for both API versions
    try:
        env = gym.make(
            'FrozenLake-v1',
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="ansi"
        )
    except TypeError:
        # Older Gym versions don't accept 'render_mode' in make()
        env = gym.make(
            'FrozenLake-v1',
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery
        )

    return env
