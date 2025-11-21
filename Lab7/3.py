import numpy as np
import random

# =========================================================
#  PART 3 — Non-Stationary 10-Armed Bandit (sample-average)
# =========================================================

class NonStationaryBandit:
    """10-arm bandit whose true rewards perform a random walk."""
    def __init__(self):
        self.k = 10
        self.q_true = np.zeros(self.k)

    def step(self):
        # random walk step (Sutton & Barto Sec 2.4)
        self.q_true += np.random.normal(0, 0.01, size=self.k)

    def pull(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.step()
        return reward


def epsilon_greedy_nonstationary(bandit, epsilon=0.1, steps=10000):
    Q = np.zeros(10)
    N = np.zeros(10)
    rewards = []

    for t in range(steps):

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 9)
        else:
            action = int(np.argmax(Q))

        reward = bandit.pull(action)

        N[action] += 1
        # BAD for nonstationary tasks (sample-average)
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    return Q, rewards


# =========================================================
# RUNNER
# =========================================================
if __name__ == "__main__":
    bandit = NonStationaryBandit()
    Q, rewards = epsilon_greedy_nonstationary(bandit)

    print("\n=== PART 3 OUTPUT ===")
    print("Final Q-values:", Q)
    print("Average reward:", sum(rewards) / len(rewards))

