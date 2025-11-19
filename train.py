import os
import pickle
import numpy as np

from flappybird_env import FlappyBirdEnv
from agents.q_learning import QAgent
from agents.sarsa import SarsaAgent
from agents.mc import MCAgent
from agents.model_base import value_iteration, policy_iteration
from utils.dataset import collect_dataset, build_model_from_dataset, evaluate_policy


def train_agent(env, agent_class, name, episodes=50000, show_every=1000):
    agent = agent_class()
    scores = []
    best_avg = 0.0

    print(f"\n=== {name.upper()} Training ({episodes} episodes) ===")

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_score = 0

        # riêng cho MC
        if isinstance(agent, MCAgent):
            agent.episode = []

        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            ep_score = info["score"]

            if isinstance(agent, QAgent):
                agent.learn(s, a, r, s2, done)
            elif isinstance(agent, SarsaAgent):
                a2 = agent.act(s2)
                agent.learn_sarsa(s, a, r, s2, a2, done)
                a = a2
            elif isinstance(agent, MCAgent):
                agent.store_transition(s, a, r)

            s = s2

        if isinstance(agent, MCAgent):
            agent.learn_episode()

        agent.decay()
        scores.append(ep_score)

        if ep % show_every == 0:
            avg = np.mean(scores[-show_every:])
            max_recent = np.max(scores[-show_every:])
            best_avg = max(best_avg, avg)
            print(
                f"  Ep {ep:5d}/{episodes} | avg: {avg:5.2f} "
                f"| max: {max_recent:2.0f} | best_avg: {best_avg:5.2f} "
                f"| eps: {agent.eps:.4f}"
            )

    agent.eps = 0.0  # Greedy for evaluation
    return agent, scores


def main():
    env = FlappyBirdEnv(render_mode=False)
    os.makedirs("results", exist_ok=True)

    # ===== Train model-free agents =====
    q_agent, q_scores = train_agent(env, QAgent, "Q-Learning")
    s_agent, s_scores = train_agent(env, SarsaAgent, "SARSA")
    mc_agent, mc_scores = train_agent(env, MCAgent, "Monte Carlo")

    # ===== Determine best agent (based on last 2000 episodes) =====
    means = [
        np.mean(q_scores[-2000:]),
        np.mean(s_scores[-2000:]),
        np.mean(mc_scores[-2000:])
    ]
    best_idx = int(np.argmax(means))
    best_agent = [q_agent, s_agent, mc_agent][best_idx]
    best_agent.eps = 0.0

    print(f"\nBest agent final mean score (last 2000 eps): {means[best_idx]:.2f}")

    # ===== Collect high-quality dataset =====
    print("\n=== Collecting dataset from best agent ===")
    dataset = collect_dataset(env, best_agent, n_episodes=5000, max_steps=2000)

    with open("results/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(
        f"✓ Dataset collected: {len(dataset)} transitions "
        f"| Avg length/ep: {len(dataset) / 5000:.1f}"
    )

    # ===== Build model for VI & PI =====
    model, states = build_model_from_dataset(dataset)
    print(f"Unique states for model-based methods: {len(states)}")

    # ===== Value Iteration =====
    print("\n=== VALUE ITERATION on learned model ===")
    V_vi, policy_vi = value_iteration(states, model)
    mean_vi, std_vi, _ = evaluate_policy(env, policy_vi)
    print(f"Value Iteration mean score: {mean_vi:.2f} ± {std_vi:.2f}")

    # ===== Policy Iteration =====
    print("\n=== POLICY ITERATION on learned model ===")
    V_pi, policy_pi = policy_iteration(states, model)
    mean_pi, std_pi, _ = evaluate_policy(env, policy_pi)
    print(f"Policy Iteration mean score: {mean_pi:.2f} ± {std_pi:.2f}")

    # ===== Save policies =====
    with open("results/policy_vi.pkl", "wb") as f:
        pickle.dump({"policy": policy_vi, "V": V_vi}, f)
    with open("results/policy_pi.pkl", "wb") as f:
        pickle.dump({"policy": policy_pi, "V": V_pi}, f)

    env.close()


if __name__ == "__main__":
    main()
