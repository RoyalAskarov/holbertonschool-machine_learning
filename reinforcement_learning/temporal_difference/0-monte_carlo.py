#!/usr/bin/env python3
"""
Monte Carlo evaluation module.
"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value estimate.

    Args:
        env: The environment instance.
        V: A numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns the next action.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V, the updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        # Calculate returns and update V (First-visit Monte Carlo)
        episode_states = [step[0] for step in episode]
        G = 0

        for i in range(len(episode) - 1, -1, -1):
            s, r = episode[i]
            G = gamma * G + r

            # Check if it is the first visit to the state in this episode
            if s not in episode_states[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return