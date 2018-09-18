# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------
import numpy as np


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def generate_noise(a_dim, curr_episode, noise_type, initial_exploration_epsilon=0.2, final_exploration_epsilon=0.2, exploration_period=200):
    if noise_type == 'OUNoise':
        noise = OUNoise(a_dim, mu=0, theta=0.15, sigma=initial_exploration_epsilon)
        if curr_episode > exploration_period:
            epsilon = final_exploration_epsilon
        else:
            epsilon = initial_exploration_epsilon - float(curr_episode) * (
                initial_exploration_epsilon - final_exploration_epsilon) / exploration_period
        noise.sigma = epsilon
        return noise.noise()
    elif noise_type == 'random':
        return (np.random.random_sample(a_dim) - 0.5) / 2 # / (1. + curr_episode)
    else:
        print "no such noise type"
        exit(0)
