import numpy as np
import random
from agents.base import BaseAgent
from utils.discretize import discretize_state


class QAgent(BaseAgent):
    def __init__(self, bins=(8, 8, 8), alpha=0.15, gamma=0.98,
                 eps=1.0, eps_min=0.01, eps_decay=0.99985):
        super().__init__(bins, gamma, eps, eps_min, eps_decay)
        self.alpha = alpha
        self.q_table = np.zeros(bins + (2,))

    def discretize(self, state):
        return discretize_state(state, self.bins)

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)
        idx = self.discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def learn(self, s, a, r, s2, done):
        i = self.discretize(s)
        j = self.discretize(s2)

        best_next = 0.0 if done else np.max(self.q_table[j])
        target = r + self.gamma * best_next

        self.q_table[i + (a,)] += self.alpha * (target - self.q_table[i + (a,)])

    def decay(self):
        super().decay()
