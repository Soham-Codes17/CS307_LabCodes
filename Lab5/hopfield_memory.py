import numpy as np
import random

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def binarize(self, pattern):
        return np.where(pattern > 0, 1, -1)

    def train(self, patterns):
        self.W[:] = 0
        for p in patterns:
            x = self.binarize(p).reshape(-1, 1)
            self.W += x @ x.T
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def recall(self, pattern, max_iters=50):
        s = self.binarize(pattern.copy())
        for _ in range(max_iters):
            prev = s.copy()
            for i in np.random.permutation(self.n):
                h_i = np.dot(self.W[i], s)
                s[i] = 1 if h_i >= 0 else -1
            if np.array_equal(prev, s):
                break
        return s

    def energy(self, s):
        return -0.5 * s.T @ self.W @ s


def generate_random_pattern(n=100):
    return np.random.choice([1, -1], size=n)


def test_capacity():
    N = 100
    hop = HopfieldNetwork(N)

    P_values = range(2, 25)
    print("\nTesting capacity...\n")

    for P in P_values:
        patterns = [generate_random_pattern(N) for _ in range(P)]
        hop.train(patterns)

        success = 0
        for p in patterns:
            recalled = hop.recall(p)
            if np.array_equal(recalled, p):
                success += 1

        print(f"P={P}, accuracy={success/P:.2f}")


def test_error_correction():
    N = 100
    P = 10   # realistic number of stored patterns
    hop = HopfieldNetwork(N)

    patterns = [generate_random_pattern(N) for _ in range(P)]
    hop.train(patterns)

    base = patterns[0]

    print("\nTesting error correction with multiple stored patterns...\n")

    for flips in [5, 10, 15, 20]:
        correct = 0
        trials = 100

        for _ in range(trials):
            corrupted = base.copy()
            idx = np.random.choice(N, flips, replace=False)
            corrupted[idx] *= -1

            recalled = hop.recall(corrupted)

            if np.array_equal(recalled, base):
                correct += 1

        print(f"P={P}, Flipped={flips}, success={correct/trials:.2f}")


if __name__ == "__main__":
    test_capacity()
    test_error_correction()
