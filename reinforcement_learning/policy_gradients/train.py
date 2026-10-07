#!/usr/bin/env python3
"""Train an agent with optional episode animation."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98,
          show_result=False):
    """
    Train an agent using Monte Carlo policy gradients.

    Args:
        env: Environment instance.
        nb_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        show_result: Whether to render every 1000 episodes.

    Returns:
        List containing the total reward for each episode.
    """
    weight = np.random.rand(
        env.observation_space.shape[0], env.action_space.n
    )
    scores = []
    base_env = env.unwrapped
    original_mode = base_env.render_mode

    try:
        for episode in range(nb_episodes):
            render_episode = show_result and episode % 1000 == 0

            if original_mode == "human":
                base_env.render_mode = (
                    "human" if render_episode else None
                )

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

                if render_episode and original_mode != "human":
                    env.render()

                if terminated or truncated:
                    break

            total_return = 0.0

            for gradient, reward in zip(
                    reversed(gradients), reversed(rewards)):
                total_return = reward + gamma * total_return
                weight += alpha * total_return * gradient

            scores.append(score)
            print("Episode: {} Score: {}".format(episode, score))
    finally:
        base_env.render_mode = original_mode

    return scores
