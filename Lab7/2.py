import numpy as np
import random

# =========================================================
#  PART 2 — Stationary Binary Bandit (ε-greedy)
# =========================================================

class BinaryBandit:
    """Two-armed stationary bandit with Bernoulli rewards."""
    def __init__(self, p1, p2):
        self.probs = [p1, p2]

    def pull(self, action):
        return 1 if random.random() < self.probs[action] else 0


def epsilon_greedy_stationary(bandit, epsilon=0.1, steps=2000):
    Q = [0.0, 0.0]    # estimated values
    N = [0, 0]        # action counts
    rewards = []

    for t in range(steps):

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q)

        reward = bandit.pull(action)

        N[action] += 1
        # sample average update
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)

    return Q, rewards


# =========================================================
# RUNNER (executes automatically)
# =========================================================
if __name__ == "__main__":
    bandit = BinaryBandit(0.8, 0.3)
    Q, rewards = epsilon_greedy_stationary(bandit)

    print("\n=== PART 2 OUTPUT ===")
    print("Final Q-values:", Q)
    print("Total reward:", sum(rewards))
    print("Average reward:", sum(rewards) / len(rewards))

