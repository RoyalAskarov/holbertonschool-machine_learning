#!/usr/bin/env python3
"""Monte Carlo algorithm module."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                 alpha=0.1, gamma=0.99):
    """
    Perform the Monte Carlo algorithm to estimate a value function.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the
            next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        episode = np.array(episode, dtype=int)
        states = episode[:, 0]
        rewards = episode[:, 1]

        G = 0
        returns = np.zeros(len(episode))
        for i in reversed(range(len(episode))):
            G = rewards[i] + gamma * G
            returns[i] = G

        visited = set()
        for i in range(len(states)):
            s = states[i]
            if s not in visited:
                visited.add(s)
                V[s] = V[s] + alpha * (returns[i] - V[s])

    return V