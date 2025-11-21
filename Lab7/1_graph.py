import matplotlib.pyplot as plt
from menace import MENACE, play_game   # <-- works after rename

if __name__ == "__main__":
    menace = MENACE()

    total_games = 5000
    window = 100

    wins = 0
    draws = 0
    losses = 0
    win_rate_curve = []

    print("Training MENACE and generating graph...")

    for i in range(total_games):
        result = play_game(menace)

        if result == "X":
            wins += 1
        elif result == "O":
            losses += 1
        else:
            draws += 1

        if (i + 1) % window == 0:
            win_rate_curve.append(wins / (i + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(win_rate_curve, label="MENACE Win Rate", linewidth=2)
    plt.title("MENACE Learning Curve")
    plt.xlabel(f"Training Progress (per {window} games)")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

