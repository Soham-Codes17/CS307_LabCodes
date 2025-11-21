import numpy as np
import random
import matplotlib.pyplot as plt

# =========================================================
# Bandit Environments
# =========================================================

class NonStationaryBandit:
    """10-armed bandit whose true values drift by random walk."""
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.zeros(k)
        self.optimal_action = 0

    def step(self):
        self.q_true += np.random.normal(0, 0.01, size=self.k)
        self.optimal_action = np.argmax(self.q_true)

    def pull(self, action):
        reward = np.random.normal(self.q_true[action], 1.0)
        self.step()
        return reward


# =========================================================
# Agents
# =========================================================

def epsilon_greedy_sample_average(bandit, epsilon=0.1, steps=10000):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)

    rewards = np.zeros(steps)
    optimal = np.zeros(steps)

    for t in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, bandit.k - 1)
        else:
            action = np.argmax(Q)

        reward = bandit.pull(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]    # sample-average update

        rewards[t] = reward
        optimal[t] = 1 if action == bandit.optimal_action else 0

    return rewards, optimal


def epsilon_greedy_constant_alpha(bandit, epsilon=0.1, alpha=0.1, steps=10000):
    Q = np.zeros(bandit.k)

    rewards = np.zeros(steps)
    optimal = np.zeros(steps)

    for t in range(steps):
        if random.random() < epsilon:
            action = random.randint(0, bandit.k - 1)
        else:
            action = np.argmax(Q)

        reward = bandit.pull(action)

        Q[action] += alpha * (reward - Q[action])  # constant step-size update

        rewards[t] = reward
        optimal[t] = 1 if action == bandit.optimal_action else 0

    return rewards, optimal


# =========================================================
# MAIN: Generate Graphs
# =========================================================
if __name__ == "__main__":
    steps = 10000
    runs = 50   # average results over multiple runs for smooth curves

    avg_reward_sample = np.zeros(steps)
    avg_optimal_sample = np.zeros(steps)

    avg_reward_alpha = np.zeros(steps)
    avg_optimal_alpha = np.zeros(steps)

    print("Running experiments...")

    for r in range(runs):
        bandit1 = NonStationaryBandit()
        bandit2 = NonStationaryBandit()

        rewards_s, optimal_s = epsilon_greedy_sample_average(bandit1, epsilon=0.1, steps=steps)
        rewards_a, optimal_a = epsilon_greedy_constant_alpha(bandit2, epsilon=0.1, alpha=0.1, steps=steps)

        avg_reward_sample += rewards_s
        avg_optimal_sample += optimal_s
        avg_reward_alpha += rewards_a
        avg_optimal_alpha += optimal_a

    avg_reward_sample /= runs
    avg_optimal_sample /= runs

    avg_reward_alpha /= runs
    avg_optimal_alpha /= runs

    # -------------------------------
    # Graph 1 – Average Reward
    # -------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(avg_reward_sample, label="Sample Average (Bad for nonstationary)")
    plt.plot(avg_reward_alpha, label="Constant Step Size α=0.1 (Good)", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Time – 10-Armed Nonstationary Bandit")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------
    # Graph 2 – % Optimal Action
    # -------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(avg_optimal_sample * 100, label="Sample Average (Bad)")
    plt.plot(avg_optimal_alpha * 100, label="Constant Step Size α=0.1 (Good)", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Optimal Action Percentage – Nonstationary Bandit")
    plt.legend()
    plt.grid(True)
    plt.show()

