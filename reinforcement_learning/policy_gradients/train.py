#!/usr/bin/env python3
"""Train an agent using Monte Carlo policy gradients."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train an agent using the REINFORCE algorithm.

    Args:
        env: Environment instance.
        nb_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        List containing the total reward for each episode.
    """
    weight = np.random.rand(
        env.observation_space.shape[0], env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        score = 0.0

        while True:
            action, gradient = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)
            score += reward

            if terminated or truncated:
                break

        total_return = 0.0

        for gradient, reward in zip(reversed(gradients), reversed(rewards)):
            total_return = reward + gamma * total_return
            weight += alpha * total_return * gradient

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores