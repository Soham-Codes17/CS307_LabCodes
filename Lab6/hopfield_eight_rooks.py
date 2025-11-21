

import numpy as np

# Board is 8x8 -> 64 neurons
BOARD_SIZE = 8
N = BOARD_SIZE * BOARD_SIZE

A = 2.0
B = 2.0

def idx(i, j):
    return i * BOARD_SIZE + j

def energy(state):
    board = state.reshape(BOARD_SIZE, BOARD_SIZE)


    row_sums = np.sum(board, axis=1)
    col_sums = np.sum(board, axis=0)

    row_term = np.sum((row_sums - 1) ** 2)


    col_term = np.sum((col_sums - 1) ** 2)

    E = A * row_term + B * col_term
    return E

def hopfield_update(state, max_iters=500):
    s = state.copy()
    for _ in range(max_iters):
        prev = s.copy()
        for i in np.random.permutation(N):
            s[i] = 1 - s[i]
            E_new = energy(s)
            s[i] = 1 - s[i]
            E_old = energy(s)

            if E_new < E_old:
                s[i] = 1 - s[i]

        if np.array_equal(prev, s):
            break
    return s

def is_valid_solution(state):
    board = state.reshape(BOARD_SIZE, BOARD_SIZE)
    row_sums = np.sum(board, axis=1)
    col_sums = np.sum(board, axis=0)
    return np.all(row_sums == 1) and np.all(col_sums == 1)

def solve_eight_rooks(restarts=50, max_iters=500):
    best_state = None
    best_energy = None

    for r in range(restarts):

        state = np.random.choice([0, 1], size=N)
        state = hopfield_update(state, max_iters=max_iters)
        E = energy(state)

        if best_energy is None or E < best_energy:
            best_energy = E
            best_state = state.copy()

        if is_valid_solution(state):
            print(f"Valid solution found at restart {r+1} with energy {E}")
            return state.reshape(BOARD_SIZE, BOARD_SIZE)

    print(f"No perfect solution found. Best energy = {best_energy}")
    return best_state.reshape(BOARD_SIZE, BOARD_SIZE)


if __name__ == "__main__":
    solution = solve_eight_rooks(restarts=50, max_iters=500)
    print("\nFinal board (1 = rook, 0 = empty):\n")
    print(solution)

    row_sums = np.sum(solution, axis=1)
    col_sums = np.sum(solution, axis=0)
    print("\nRow sums:", row_sums)
    print("Column sums:", col_sums)
