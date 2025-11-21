import numpy as np
import random

# =========================================================
#  PART 4 — Modified ε-greedy (constant α)
# =========================================================

class NonStationaryBandit:
    """10-arm bandit with random-walk reward means."""
    def __init__(self):
        self.k = 10
        self.q_true = np.zeros(self.k)

    def step(self):
        self.q_true += np.random.normal(0, 0.01, size=self.k)

    def pull(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.step()
        return reward


def modified_epsilon_greedy(bandit, epsilon=0.1, alpha=0.1, steps=10000):
    Q = np.zeros(10)
    rewards = []

    for t in range(steps):

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 9)
        else:
            action = int(np.argmax(Q))

        reward = bandit.pull(action)

        # constant step size update (Sutton Eq. 2.6)
        Q[action] += alpha * (reward - Q[action])

        rewards.append(reward)

    return Q, rewards


# =========================================================
# RUNNER
# =========================================================
if __name__ == "__main__":
    bandit = NonStationaryBandit()
    Q, rewards = modified_epsilon_greedy(bandit)

    print("\n=== PART 4 OUTPUT ===")
    print("Final Q-values:", Q)
    print("Average reward:", sum(rewards) / len(rewards))

