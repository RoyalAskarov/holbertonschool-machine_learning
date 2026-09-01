#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm."""
    for _ in range(episodes):
        state, _ = env.reset()
        states = []
        rewards = []

        for _ in range(max_steps):
            states.append(state)

            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        for t in range(len(states)):
            G = 0

            for k in range(t, len(rewards)):
                G += (gamma ** (k - t)) * rewards[k]

            state = states[t]
            V[state] = (1 - alpha) * V[state] + alpha * G

    return V