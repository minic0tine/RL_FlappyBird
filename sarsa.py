from agents.q_learning import QAgent


class SarsaAgent(QAgent):
    def learn_sarsa(self, s, a, r, s2, a2, done):
        i = self.discretize(s)
        j = self.discretize(s2)

        q_next = 0.0 if done else self.q_table[j + (a2,)]
        target = r + self.gamma * q_next

        self.q_table[i + (a,)] += self.alpha * (target - self.q_table[i + (a,)])
