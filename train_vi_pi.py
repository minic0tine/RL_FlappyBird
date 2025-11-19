import os
import pickle
import time

from flappybird_env import FlappyBirdEnv
from utils.dataset import build_model_from_dataset, evaluate_policy
from agents.model_base import value_iteration, policy_iteration, PolicyAgent


def main():
    print("\n==============================")
    print(" VALUE ITERATION & POLICY ITERATION")
    print("==============================")

    env = FlappyBirdEnv(render_mode=False)
    os.makedirs('results', exist_ok=True)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    dataset_path = 'results/dataset.pkl'
    if not os.path.exists(dataset_path):
        print("Dataset not found! Run train.py first to collect dataset.")
        return

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset: {len(dataset)} transitions")

    # 2. Build model
    print("\n[2/5] Building MDP model...")
    model, states = build_model_from_dataset(dataset)

    print(f"Total states: {len(states)}")
    print(f"State-action pairs: {len(model.P)}")
    print(f"Avg actions/state: {len(model.P) / len(states):.2f}")

    # 3. VALUE ITERATION
    print("\n[3/5] Running Value Iteration...")
    start = time.time()
    V_vi, policy_vi = value_iteration(states, model, gamma=0.98, iters=300, tol=1e-4)
    vi_time = time.time() - start

    print(f"Value Iteration done in {vi_time:.2f}s")

    vi_agent = PolicyAgent(policy_vi, bins=(8, 8, 8))
    mean_vi, std_vi, scores_vi = evaluate_policy(env, vi_agent, episodes=100)

    # 4. POLICY ITERATION
    print("\n[4/5] Running Policy Iteration...")
    start = time.time()
    V_pi, policy_pi = policy_iteration(states, model, gamma=0.98, eval_iters=60, max_iters=100)
    pi_time = time.time() - start

    print(f"Policy Iteration done in {pi_time:.2f}s")

    pi_agent = PolicyAgent(policy_pi, bins=(8, 8, 8))
    mean_pi, std_pi, scores_pi = evaluate_policy(env, pi_agent, episodes=100)

    # 5. Summary
    print("\n==============================")
    print(" COMPARISON")
    print("==============================")
    print(f"VI mean score: {mean_vi:.2f} ± {std_vi:.2f}")
    print(f"PI mean score: {mean_pi:.2f} ± {std_pi:.2f}")
    print(f"VI time: {vi_time:.2f}s | PI time: {pi_time:.2f}s")

    summary = {
        'vi': {'mean': mean_vi, 'std': std_vi, 'scores': scores_vi},
        'pi': {'mean': mean_pi, 'std': std_pi, 'scores': scores_pi}
    }
    with open('results/vi_pi_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print("\nSaved summary → results/vi_pi_summary.pkl")


if __name__ == "__main__":
    main()
