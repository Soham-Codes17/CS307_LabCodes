import numpy as np
import random
import matplotlib.pyplot as plt

# =============================================================
#       STATIONARY 2-ARMED BANDIT GRAPH (Part 2)
# =============================================================

class BinaryBandit:
    def __init__(self, p1, p2):
        self.probs = [p1, p2]

    def pull(self, action):
        return 1 if random.random() < self.probs[action] else 0


def epsilon_greedy_stationary(bandit, epsilon=0.1, steps=2000):
    Q = [0.0, 0.0]
    N = [0, 0]
    rewards = []

    for t in range(steps):
        # epsilon greedy
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q)

        reward = bandit.pull(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]   # sample-average

        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    steps = 2000
    bandit = BinaryBandit(0.8, 0.3)

    print("Running Part 2 graph generation...")

    rewards = epsilon_greedy_stationary(bandit, epsilon=0.1, steps=steps)

    # Moving average for smoother graph
    window = 50
    smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10,5))
    plt.plot(smooth, label="Average Reward", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Reward (moving avg)")
    plt.title("Stationary 2-Armed Bandit – Average Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

