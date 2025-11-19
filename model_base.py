import random
import numpy as np
from collections import defaultdict
from utils.discretize import discretize_state


class LearnedModel:
    """
    Learned tabular MDP model:
      transitions[(s, a)] -> counts of s'
      rewards[(s, a)] -> list of rewards
      dones[(s, a)]   -> list of done flags
    """
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.rewards = defaultdict(list)
        self.dones = defaultdict(list)

    def add(self, s, a, s2, r, done=False):
        key = (s, a)
        self.transitions[key][s2] += 1
        self.rewards[key].append(r)
        self.dones[key].append(1.0 if done else 0.0)

    def build(self):
        """
        Build probability and reward tables:
        P[(s,a)] = {'s_next': {s2: prob}, 'r': mean_reward, 'done': prob_done}
        R[(s,a)] = mean_reward
        """
        self.P = {}
        self.R = {}

        for key, counter in self.transitions.items():
            total = sum(counter.values())
            self.P[key] = {
                's_next': {s2: cnt / total for s2, cnt in counter.items()},
                'r': float(np.mean(self.rewards[key])),
                'done': float(np.mean(self.dones[key]))
            }
            self.R[key] = self.P[key]['r']

    def get_transitions(self, s, a):
        if (s, a) in self.P:
            return self.P[(s, a)]['s_next']
        return {}

    def get_reward(self, s, a):
        return self.R.get((s, a), 0.0)

    def get_done_prob(self, s, a):
        if (s, a) in self.P:
            return self.P[(s, a)]['done']
        return 0.0


def value_iteration(states, model, gamma=0.98, iters=300, tol=1e-4):
    """
    Standard value iteration on learned model.
    """
    V = {s: 0.0 for s in states}

    print(f"   [VI] Running with {len(states)} states...")

    for it in range(iters):
        delta = 0.0

        for s in states:
            v_old = V[s]
            best = -1e9

            for a in [0, 1]:
                if (s, a) not in model.P:
                    continue

                trans = model.P[(s, a)]
                r = trans['r']
                done_prob = trans['done']

                val = r
                if done_prob < 0.5:  # mostly non-terminal
                    val += gamma * sum(p * V.get(s2, 0.0) for s2, p in trans['s_next'].items())

                if val > best:
                    best = val

            if best > -1e9:
                V[s] = best

            delta = max(delta, abs(V[s] - v_old))

        if it % 50 == 0:
            print(f"      Iter {it}: delta={delta:.6f}, V_mean={np.mean(list(V.values())):.3f}")

        if delta < tol:
            print(f"   [VI] Converged at iter {it}")
            break

    # Extract greedy policy
    policy = {}
    for s in states:
        best_a, best_v = 0, -1e9

        for a in [0, 1]:
            if (s, a) not in model.P:
                continue

            trans = model.P[(s, a)]
            r = trans['r']
            done_prob = trans['done']

            val = r
            if done_prob < 0.5:
                val += gamma * sum(p * V.get(s2, 0.0) for s2, p in trans['s_next'].items())

            if val > best_v:
                best_a, best_v = a, val

        policy[s] = best_a

    return V, policy


def policy_iteration(states, model, gamma=0.98, eval_iters=60, max_iters=100):
    """
    Standard policy iteration on learned model.
    """
    policy = {s: random.randint(0, 1) for s in states}
    V = {s: 0.0 for s in states}

    print(f"   [PI] Running with {len(states)} states...")

    for k in range(max_iters):
        # Policy Evaluation
        for _ in range(eval_iters):
            for s in states:
                if s not in policy:
                    continue

                a = policy[s]

                if (s, a) not in model.P:
                    V[s] = model.get_reward(s, a)
                    continue

                trans = model.P[(s, a)]
                r = trans['r']
                done_prob = trans['done']

                V[s] = r
                if done_prob < 0.5:
                    V[s] += gamma * sum(p * V.get(s2, 0.0) for s2, p in trans['s_next'].items())

        # Policy Improvement
        stable = True

        for s in states:
            old_a = policy[s]
            best_a, best_v = 0, -1e9

            for a in [0, 1]:
                if (s, a) not in model.P:
                    continue

                trans = model.P[(s, a)]
                r = trans['r']
                done_prob = trans['done']

                val = r
                if done_prob < 0.5:
                    val += gamma * sum(p * V.get(s2, 0.0) for s2, p in trans['s_next'].items())

                if val > best_v:
                    best_a, best_v = a, val

            policy[s] = best_a
            if policy[s] != old_a:
                stable = False

        if k % 20 == 0:
            print(f"      Iter {k}: V_mean={np.mean(list(V.values())):.3f}")

        if stable:
            print(f"   [PI] Converged at iter {k}")
            break

    return V, policy


class PolicyAgent:
    """
    Wrap a tabular policy (dict s_disc -> action) into an agent with .act().
    """
    def __init__(self, policy, bins=(8, 8, 8)):
        self.policy = policy
        self.bins = bins
        self.eps = 0.0  # purely greedy

    def act(self, state):
        s_disc = discretize_state(state, self.bins)
        return self.policy.get(s_disc, 0)
