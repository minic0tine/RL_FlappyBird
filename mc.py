import numpy as np
import random
from collections import defaultdict
from utils.discretize import discretize_state


def zeros_array():
    return np.zeros(2)


def zeros_int_array():
    return np.zeros(2, dtype=int)


class MCAgent:
    def __init__(self, bins=(8, 8, 8), gamma=0.98,
                 eps=1.0, eps_min=0.01, eps_decay=0.99985):
        self.bins = bins
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # Q[state_bins][action]
        self.Q = np.zeros(bins + (2,))
        self.returns_sum = defaultdict(zeros_array)
        self.returns_count = defaultdict(zeros_int_array)
        self.episode = []

    def _disc(self, state):
        return discretize_state(state, self.bins)

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)
        idx = self._disc(state)
        return int(np.argmax(self.Q[idx]))

    def store_transition(self, state, action, reward):
        self.episode.append((state, action, reward))

    def learn_episode(self):
        """
        First-visit Monte Carlo: update Q for each (state, action) first visited in episode.
        """
        G = 0.0
        visited = set()

        for state, action, reward in reversed(self.episode):
            idx = self._disc(state)
            key = idx + (action,)

            if key not in visited:
                visited.add(key)
                G = reward + self.gamma * G

                self.returns_sum[idx][action] += G
                self.returns_count[idx][action] += 1

                self.Q[idx][action] = (
                    self.returns_sum[idx][action] / self.returns_count[idx][action]
                )

        self.episode = []

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
