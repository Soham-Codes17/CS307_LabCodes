from collections import deque
import time

# ---------- Missionaries and Cannibals ----------

def is_valid_state(m_left, c_left):
    """Check constraints validity."""
    if m_left < 0 or c_left < 0 or m_left > 3 or c_left > 3:
        return False
    m_right, c_right = 3 - m_left, 3 - c_left
    if (m_left > 0 and m_left < c_left) or (m_right > 0 and m_right < c_right):
        return False
    return True


def get_successors(state):
    m_left, c_left, boat = state
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    successors = []
    for m, c in moves:
        if boat == 'L':
            new_state = (m_left - m, c_left - c, 'R')
        else:
            new_state = (m_left + m, c_left + c, 'L')
        if is_valid_state(*new_state[:2]):
            successors.append(new_state)
    return successors


def bfs_missionaries():
    start = (3, 3, 'L')
    goal = (0, 0, 'R')
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        state = path[-1]
        if state == goal:
            return path
        for succ in get_successors(state):
            if succ not in visited:
                visited.add(succ)
                queue.append(path + [succ])
    return None


def dfs_missionaries():
    start = (3, 3, 'L')
    goal = (0, 0, 'R')
    stack = [[start]]
    visited = {start}

    while stack:
        path = stack.pop()
        state = path[-1]
        if state == goal:
            return path
        for succ in get_successors(state):
            if succ not in visited:
                visited.add(succ)
                stack.append(path + [succ])
    return None


# ---------- Rabbit Leap ----------

def get_rabbit_successors(state):
    successors = []
    s = list(state)
    for i, r in enumerate(s):
        if r == 'E':
            # Move east
            if i + 1 < len(s) and s[i + 1] == '_':
                t = s.copy()
                t[i], t[i + 1] = t[i + 1], t[i]
                successors.append(tuple(t))
            elif i + 2 < len(s) and s[i + 1] in ('E', 'W') and s[i + 2] == '_':
                t = s.copy()
                t[i], t[i + 2] = t[i + 2], t[i]
                successors.append(tuple(t))
        elif r == 'W':
            # Move west
            if i - 1 >= 0 and s[i - 1] == '_':
                t = s.copy()
                t[i], t[i - 1] = t[i - 1], t[i]
                successors.append(tuple(t))
            elif i - 2 >= 0 and s[i - 1] in ('E', 'W') and s[i - 2] == '_':
                t = s.copy()
                t[i], t[i - 2] = t[i - 2], t[i]
                successors.append(tuple(t))
    return successors


def bfs_rabbits():
    start = tuple(['E', 'E', 'E', '_', 'W', 'W', 'W'])
    goal = tuple(['W', 'W', 'W', '_', 'E', 'E', 'E'])
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        state = path[-1]
        if state == goal:
            return path
        for succ in get_rabbit_successors(state):
            if succ not in visited:
                visited.add(succ)
                queue.append(path + [succ])
    return None


def dfs_rabbits():
    start = tuple(['E', 'E', 'E', '_', 'W', 'W', 'W'])
    goal = tuple(['W', 'W', 'W', '_', 'E', 'E', 'E'])
    stack = [[start]]
    visited = {start}

    while stack:
        path = stack.pop()
        state = path[-1]
        if state == goal:
            return path
        for succ in get_rabbit_successors(state):
            if succ not in visited:
                visited.add(succ)
                stack.append(path + [succ])
    return None


# ---------- Compare BFS & DFS ----------

def compare_searches():
    print("Missionaries & Cannibals:")
    t1 = time.time()
    bfs_path = bfs_missionaries()
    t2 = time.time()
    dfs_path = dfs_missionaries()
    t3 = time.time()

    print(f"BFS solution ({len(bfs_path)-1} steps): {bfs_path}")
    print(f"DFS solution ({len(dfs_path)-1} steps): {dfs_path}")
    print(f"BFS time: {t2-t1:.5f}s | DFS time: {t3-t2:.5f}s\n")

    print("Rabbit Leap:")
    t1 = time.time()
    bfs_path = bfs_rabbits()
    t2 = time.time()
    dfs_path = dfs_rabbits()
    t3 = time.time()

    print(f"BFS solution ({len(bfs_path)-1} steps):")
    for step in bfs_path:
        print(''.join(step))
    print(f"DFS solution ({len(dfs_path)-1} steps):")
    for step in dfs_path:
        print(''.join(step))
    print(f"BFS time: {t2-t1:.5f}s | DFS time: {t3-t2:.5f}s")


if __name__ == "__main__":
    compare_searches()
