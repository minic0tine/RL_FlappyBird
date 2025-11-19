class BaseAgent:
    def __init__(self, bins=(8, 8, 8), gamma=0.98, eps=1.0, eps_min=0.01, eps_decay=0.99985):
        self.bins = bins
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def decay(self):
        """Decay epsilon for epsilon-greedy."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
