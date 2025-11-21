"""
MENACE – Matchbox Educable Noughts And Crosses Engine
Faithful reconstruction based on:
Michie, D. “Experiments on the mechanization of game-learning” (1963)
MATCHBOX paper provided by user.
"""

import random
import itertools
import copy

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def rotate(board):
    """Rotate board 90 degrees clockwise"""
    return [
        board[6], board[3], board[0],
        board[7], board[4], board[1],
        board[8], board[5], board[2]
    ]

def reflect(board):
    """Reflect board horizontally"""
    return [
        board[2], board[1], board[0],
        board[5], board[4], board[3],
        board[8], board[7], board[6]
    ]

def canonical(board):
    """Return the canonical form: minimal among all rotations/reflections"""
    boards = []
    b = board[:]
    for _ in range(4):
        boards.append(b)
        boards.append(reflect(b))
        b = rotate(b)
    return min(boards)

# ---------------------------------------------------------------
# Bead Color Mapping (Table 1 in the paper)
# ---------------------------------------------------------------

COLOR_CODE = {
    1: "WHITE",
    2: "LILAC",
    3: "SILVER",
    4: "GREEN",
    5: "PINK",
    6: "RED",
    7: "AMBER",
    8: "BLACK",
    0: "GOLD"
}

# Michie uses numbering:
# 1 2 3
# 8 0 4
# 7 6 5

CONVERSION = [1,2,3,
              8,0,4,
              7,6,5]

# ---------------------------------------------------------------
# MENACE Class
# ---------------------------------------------------------------

class MENACE:
    def __init__(self):
        # matchbox: key = canonical board state → value = {moveIndex: beadCount}
        self.matchboxes = {}
        self.history = []

    def possible_moves(self, board):
        return [i for i,x in enumerate(board) if x == "-"]

    def to_michie_index(self, move):
        """Convert 0–8 index → Michie numbering system"""
        return CONVERSION[move]

    def initial_beads_for_box(self, moves):
        """Table 2 bead replication depending on stage of play"""
        # Stage unknown initially; use default replication 4 beads
        beads = {}
        for m in moves:
            beads[m] = 4   # stronger initial bias (matches Table 2 stage 1)
        return beads

    def get_box(self, board):
        c = tuple(canonical(board))
        if c not in self.matchboxes:
            moves = self.possible_moves(c)
            self.matchboxes[c] = self.initial_beads_for_box(moves)
        return c, self.matchboxes[c]

    def choose_move(self, board):
        box_state, box = self.get_box(board)
        moves = list(box.keys())
        bead_counts = list(box.values())

        move = random.choices(moves, weights=bead_counts)[0]
        self.history.append((box_state, move))
        return move

    # -----------------------------------------------------------
    # Reinforcement rules from paper:
    # WIN  = +3 beads each
    # DRAW = +1 beads each
    # LOSS = -1 bead (if possible)
    # Bead change depends on stage of play (Table 2)
    # Later moves get stronger reinforcement
    # -----------------------------------------------------------

    def reinforce(self, result):
        # result = 1 win, 0 draw, -1 loss
        n = len(self.history)

        for stage, (state, move) in enumerate(self.history):
            # moves closer to end should get stronger reinforcement
            # Using Table 2 explicitly:
            # stage from end: lastmove=1→4 beads, earlier→3,2,1
            distance_from_end = n - stage
            if distance_from_end == 1:
                weight = 4
            elif distance_from_end == 2:
                weight = 3
            elif distance_from_end == 3:
                weight = 2
            else:
                weight = 1

            if result == 1:        # WIN
                self.matchboxes[state][move] += 3 * weight
            elif result == 0:      # DRAW
                self.matchboxes[state][move] += 1 * weight
            else:                  # LOSS
                self.matchboxes[state][move] = max(1, self.matchboxes[state][move] - 1 * weight)

        self.history = []

# ---------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------

def check_winner(b):
    wins = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for (i,j,k) in wins:
        if b[i] == b[j] == b[k] and b[i] != "-":
            return b[i]
    if "-" not in b:
        return "D"
    return None

def play_game(menace, opponent="random"):
    board = ["-"]*9
    turn = "X"   # MENACE plays X

    while True:
        if turn == "X":
            move = menace.choose_move(board)
        else:
            legal = [i for i in range(9) if board[i] == "-"]
            move = random.choice(legal)

        board[move] = turn
        result = check_winner(board)

        if result:
            if result == "X": menace.reinforce(+1)
            elif result == "O": menace.reinforce(-1)
            else: menace.reinforce(0)
            return result

        turn = "O" if turn == "X" else "X"

# ---------------------------------------------------------------
# TRAINING RUNNER (INCORPORATED INTO MAIN CODE)
# ---------------------------------------------------------------

if __name__ == "__main__":
    menace = MENACE()

    wins = draws = losses = 0

    # Train MENACE for 5000 games
    for _ in range(5000):
        result = play_game(menace)

        if result == "X":
            wins += 1
        elif result == "O":
            losses += 1
        else:
            draws += 1

    print("Training complete!")
    print(f"Wins:   {wins}")
    print(f"Draws:  {draws}")
    print(f"Losses: {losses}")

