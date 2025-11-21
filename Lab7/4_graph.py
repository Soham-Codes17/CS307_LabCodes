import numpy as np
import random
import matplotlib.pyplot as plt

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
    optimal_actions = []

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

        # track optimal action
        optimal_action = np.argmax(bandit.q_true)
        optimal_actions.append(1 if action == optimal_action else 0)

    return rewards, optimal_actions


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Generating graphs for Part 4...")

    steps = 10000
    runs = 30

    avg_rewards = np.zeros(steps)
    avg_optimal = np.zeros(steps)

    for r in range(runs):
        bandit = NonStationaryBandit()
        rewards, optimal = modified_epsilon_greedy(bandit)

        avg_rewards += np.array(rewards)
        avg_optimal += np.array(optimal)

    avg_rewards /= runs
    avg_optimal = (avg_optimal / runs) * 100

    # ----------------------------
    # GRAPH 1: Average Reward (Part 4)
    # ----------------------------
    plt.figure(figsize=(10,5))
    plt.plot(avg_rewards, linewidth=2)
    plt.title("Part 4: Average Reward Over Time (Constant Step-Size)")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig("5_graph.png", dpi=300)
    plt.show()

    # ----------------------------
    # GRAPH 2: Optimal Action Percentage (Part 4)
    # ----------------------------
    plt.figure(figsize=(10,5))
    plt.plot(avg_optimal, linewidth=2)
    plt.title("Part 4: Optimal Action Percentage (Constant Step-Size)")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.grid(True)
    plt.savefig("6_graph.png", dpi=300)
    plt.show()

    print("Graphs saved as 5_graph.png and 6_graph.png")

