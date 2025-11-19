import numpy as np
import random
from utils.discretize import discretize_state


def collect_dataset(env, agent=None, n_episodes=5000, max_steps=2000):
    """
    Collect dataset of transitions using discretize_state.
    Returns: list of (s_disc, a, s2_disc, r, done)
    """
    from statistics import mean

    dataset = []
    scores = []

    bins = getattr(agent, 'bins', (8, 8, 8))

    for ep in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            if agent is not None:
                a = agent.act(s)
            else:
                a = random.randint(0, 1)

            s2, r, done, info = env.step(a)

            s_disc = discretize_state(s, bins)
            s2_disc = discretize_state(s2, bins)

            dataset.append((s_disc, a, s2_disc, r, done))

            s = s2
            steps += 1

        scores.append(info['score'])

        if (ep + 1) % 500 == 0:
            recent_avg = mean(scores[-500:])
            print(f"   Collected {ep+1}/{n_episodes}, recent avg score: {recent_avg:.2f}")

    return dataset


def build_model_from_dataset(dataset):
    """
    Build a learned MDP model from dataset.
    """
    from collections import defaultdict
    from agents.model_base import LearnedModel

    model = LearnedModel()
    states = set()

    for item in dataset:
        if len(item) == 5:
            s, a, s2, r, done = item
        else:
            s, a, s2, r = item
            done = False

        model.add(s, a, s2, r, done)
        states.add(s)
        states.add(s2)

    model.build()

    print(f"   Model built: {len(states)} states, {len(model.P)} state-action entries")
    return model, list(states)


def evaluate_policy(env, policy, episodes=100, bins=(8, 8, 8)):
    """
    Evaluate a deterministic policy or an agent object on the environment.
    - If policy is dict: key is discrete state, value is action.
    - If policy is an agent: must have .act(state).
    """
    from utils.discretize import discretize_state

    scores = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        steps = 0

        while not done and steps < 3000:
            s_disc = discretize_state(s, bins)

            if isinstance(policy, dict):
                a = policy.get(s_disc, 0)
            else:
                a = policy.act(s)

            s, r, done, info = env.step(a)
            steps += 1

        scores.append(info["score"])

    return np.mean(scores), np.std(scores), scores
