
# 🐦 Reinforcement Learning on Flappy Bird  
**A study of model-free and model-based RL algorithms on a custom Flappy Bird environment.**

This project implements and compares five classical Reinforcement Learning (RL) algorithms on a custom Pygame-based Flappy Bird environment. The environment is discretized to support tabular RL and includes reward shaping to stabilize training.

---

## 📌 **Algorithms Implemented**

### ✅ Model-Free Methods
- **Q-Learning**
- **SARSA**
- **Monte Carlo Control (First-Visit)**

### ✅ Model-Based Methods
- **Value Iteration**
- **Policy Iteration**

Each algorithm is evaluated based on:
- Final performance (average return over last 50 episodes)
- Sensitivity to hyperparameters:
  - Learning rate (α)
  - Exploration rate (ε)
  - Discount factor (γ)

---

## 🏗 **Project Structure**

