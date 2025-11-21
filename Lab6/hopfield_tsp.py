import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN_AVAILABLE = True
except Exception:
    _HUNGARIAN_AVAILABLE = False

CITIES = 10
N = CITIES * CITIES

A = 500.0
B = 500.0
C = 1.0
u0 = 0.02
dt = 0.01
ITERATIONS = 20000
RESTARTS = 8
NOISE_STD = 0.002
PRINT_EVERY = 2000
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)

def make_random_distance_matrix(n, low=5, high=50, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    D = rng.integers(low, high, size=(n, n)).astype(float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D

def activation(u):
    return 0.5 * (1.0 + np.tanh(u / u0))

def greedy_project_to_permutation(x):
    n = x.shape[0]
    scores = x.copy()
    perm = -np.ones(n, dtype=int)
    USED = -np.inf
    for _ in range(n):
        idx = np.unravel_index(np.nanargmax(scores), scores.shape)
        i, a = idx
        if scores[i, a] == USED:
            break
        perm[a] = int(i)
        scores[i, :] = USED
        scores[:, a] = USED
    for col in range(n):
        if perm[col] == -1:
            perm[col] = int(np.argmax(x[:, col]))
    return perm.tolist()

def hungarian_project_to_permutation(x):
    cost = -x
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = -np.ones(x.shape[1], dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = int(r)
    return perm.tolist()

def project_to_permutation(x):
    if _HUNGARIAN_AVAILABLE:
        return hungarian_project_to_permutation(x)
    else:
        return greedy_project_to_permutation(x)

def permutation_to_onehot(perm):
    n = len(perm)
    mat = np.zeros((n, n), dtype=int)
    for pos, city in enumerate(perm):
        mat[city, pos] = 1
    return mat

def tour_cost(tour, D):
    n = len(tour)
    return sum(D[tour[i], tour[(i + 1) % n]] for i in range(n))

def run_hopfield_once(D, params, rng):
    n = D.shape[0]
    A = params['A']; B = params['B']; C = params['C']
    dt = params['dt']; iterations = params['iterations']
    noise_std = params['noise_std']; print_every = params['print_every']

    u = rng.normal(loc=0.0, scale=0.5, size=(n, n))

    for step in range(iterations):
        x = activation(u)
        row_sums = x.sum(axis=1)
        col_sums = x.sum(axis=0)
        x_next = np.roll(x, -1, axis=1)
        x_prev = np.roll(x, 1, axis=1)
        neighbor_sum = x_next + x_prev
        term = D.dot(neighbor_sum)
        du = -u - A * (row_sums[:, None] - 1.0) - B * (col_sums[None, :] - 1.0) - C * term
        u += dt * du + rng.normal(0.0, noise_std, size=(n, n))
        if (step % print_every) == 0:
            print(f"iter {step}/{iterations}")

    x_final = activation(u)
    perm = project_to_permutation(x_final)
    onehot = permutation_to_onehot(perm)
    return perm, onehot, x_final

if __name__ == "__main__":
    D = make_random_distance_matrix(CITIES, rng=rng)

    params = {
        'A': A, 'B': B, 'C': C,
        'dt': dt, 'iterations': ITERATIONS,
        'noise_std': NOISE_STD, 'print_every': PRINT_EVERY
    }

    best_cost = float('inf')
    best_perm = None
    best_onehot = None
    best_x_final = None

    print("\nSolving TSP using corrected Hopfield-Tank dynamics...\n")
    print(f"Hungarian available: {_HUNGARIAN_AVAILABLE}\n")

    for r in range(RESTARTS):
        print(f"--- Restart {r + 1}/{RESTARTS} ---")
        perm, onehot, x_final = run_hopfield_once(D, params, rng)
        cost = tour_cost(perm, D)
        print(f"Restart {r + 1} cost = {cost}\n")
        if cost < best_cost:
            best_cost = cost
            best_perm = perm[:]
            best_onehot = onehot.copy()
            best_x_final = x_final.copy()
            print(f"  -> New best found (cost {best_cost})\n")

    if best_perm is None:
        print("No valid tour found in restarts.")
    else:
        print("Best tour (position 0..n-1 -> city at that position):")
        print(best_perm)
        print("\nProjected one-hot tour matrix (rows=cities, cols=positions):")
        np.set_printoptions(linewidth=160, precision=3, suppress=True)
        print(best_onehot)
        print("\nBest tour cost:", best_cost)

    print(f"\nTotal neurons: {N}; Total unique symmetric weights (undirected, no self): { (N * (N - 1)) // 2 }")
