# raag_bhairav_ga.py
import numpy as np
import random
from scipy.io import wavfile

# ---------------------------
# Config & musical params
# ---------------------------
SR = 22050               # sample rate for audio
DURATION_SEC = 12        # (approx) length target (used for scaling)
POPULATION = 120
GENERATIONS = 250
ELITE_K = 6
MUTATION_RATE = 0.18
CROSSOVER_RATE = 0.9
MELODY_LEN = 24          # number of notes per melody

# Bhairav scale semitone offsets from tonic (Sa) in one octave:
# Sa, komal Re, Ga, Ma, Pa, komal Dha, Ni
BHAIRAV_ONE_OCT = [0, 1, 4, 5, 7, 8, 11]

# Expand to two octaves (so we allow Sa..Sa' and some above/below)
BHAIRAV = BHAIRAV_ONE_OCT + [x + 12 for x in BHAIRAV_ONE_OCT]  # 14 notes

# Map scale degrees to MIDI-like frequencies (we'll choose a tonic frequency)
TONIC_FREQ = 261.6256  # Middle C ~ C4 as tonic (Sa). Choose any frequency you like.

def semitone_to_freq(semitones_from_sa):
    return TONIC_FREQ * (2 ** (semitones_from_sa / 12.0))

# duration choices (in beats); we'll normalize to a total audio length later
DURATIONS = [0.25, 0.5, 0.75, 1.0]  # sixteenth, eighth, dotted-eighth, quarter

# motifs (in semitone offsets, relative to scale indices). We'll encode motifs as sequences of scale-index offsets
# We'll define motifs directly as scale indices relative to BHAIRAV list positions:
MOTIFS = [
    [0, 1, 2],        # Sa Re Ga (lower octave indices: 0->Sa, 1->komal Re, 2->Ga)
    [2, 1, 0],        # Ga Re Sa
    [11, 13, 14],     # Komal Dha (11), Ni (13), Sa' (14) relative indices in BHAIRAV (we'll match more flexibly)
    [0,1,2,3,4]       # Sa Re Ga Ma Pa (ascending phrase)
]

# ---------------------------
# Individual representation
# ---------------------------
# Individual: list of tuples (scale_index, duration_index)
SCALE_INDICES = list(range(len(BHAIRAV)))

def random_individual():
    return [(random.choice(SCALE_INDICES), random.randrange(len(DURATIONS))) for _ in range(MELODY_LEN)]

# ---------------------------
# Fitness function
# ---------------------------
def fitness(ind):
    """
    Higher is better.
    Components:
    - motif_score: reward occurrences of motifs (sliding window, tolerant match).
    - smoothness_score: penalize large jumps (in semitones).
    - ending_score: reward ending on tonic (Sa) or cadence.
    - rhythmic variety score: prefer some variation in durations.
    - pitch variety (not too repetitive).
    """
    # decode pitch semitone values
    semis = [BHAIRAV[p] for p, d in ind]
    durs = [DURATIONS[d] for p,d in ind]

    # motif score: check if any motif appears as subsequence in semitone differences or scale-index pattern
    motif_score = 0.0
    for motif in MOTIFS:
        # match motif against scale-index sequences (sliding window)
        for i in range(len(ind) - len(motif) + 1):
            window_idxs = [ind[i + k][0] % len(BHAIRAV) for k in range(len(motif))]
            # simple equality match on indices (allow octave shifts)
            base = window_idxs[0]
            shifted = [(x - base) % len(BHAIRAV) for x in window_idxs]
            if shifted == motif[:len(shifted)]:
                motif_score += 3.0

    # smoothness: penalize big leaps in semitones
    leaps = [abs(semis[i+1] - semis[i]) for i in range(len(semis)-1)]
    leap_penalty = sum([max(0, (l - 7))**1.2 for l in leaps])  # penalize leaps > 7 semitones strongly

    # ending: reward last note being tonic (Sa in middle or Sa' i.e., semitone 0 mod 12)
    last = semis[-1] % 12
    ending_score = 5.0 if last == 0 else 0.0

    # rhythmic variety: entropy-like measure of durations
    unique_durs = len(set(durs))
    duration_score = unique_durs * 0.5

    # pitch variety: encourage some variety but not random noise
    unique_pitches = len(set([s % 12 for s in semis]))
    pitch_score = min(unique_pitches, 8) * 0.4

    # melodic contour: reward small average jump
    avg_leap = (sum(leaps)/len(leaps)) if leaps else 0
    smoothness_score = max(0, 6 - avg_leap)  # bigger when avg_leap small

    score =  motif_score + duration_score + pitch_score + smoothness_score + ending_score - (0.3 * leap_penalty)
    return score

# ---------------------------
# GA operators
# ---------------------------
def crossover(a, b):
    if random.random() > CROSSOVER_RATE:
        return a.copy(), b.copy()
    # one-point crossover
    pt = random.randrange(1, MELODY_LEN-1)
    child1 = a[:pt] + b[pt:]
    child2 = b[:pt] + a[pt:]
    return child1, child2

def mutate(ind):
    new = [x for x in ind]
    for i in range(len(new)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.6:
                # mutate pitch
                new[i] = (random.choice(SCALE_INDICES), new[i][1])
            else:
                # mutate duration
                new[i] = (new[i][0], random.randrange(len(DURATIONS)))
    return new

# ---------------------------
# GA main loop
# ---------------------------
def run_ga():
    pop = [random_individual() for _ in range(POPULATION)]
    best = None
    best_score = -1e9

    for gen in range(GENERATIONS):
        scored = [(fitness(ind), ind) for ind in pop]
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best = scored[0][1]
        if gen % 10 == 0:
            print(f"Gen {gen}: Best fitness {scored[0][0]:.3f}")

        # Elitism
        newpop = [ind for (_, ind) in scored[:ELITE_K]]

        # selection: roulette wheel on positive fitness
        min_score = min([s for s,i in scored])
        adj = [(s - min_score + 1e-6) for s,i in scored]  # make positive
        total = sum(adj)
        probs = [a/total for a in adj]

        # helper to pick index
        def pick_index():
            r = random.random()
            cum = 0.0
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    return idx
            return len(probs)-1

        # create children
        while len(newpop) < POPULATION:
            i1 = pick_index()
            i2 = pick_index()
            p1 = scored[i1][1]
            p2 = scored[i2][1]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            newpop.append(c1)
            if len(newpop) < POPULATION:
                newpop.append(c2)

        pop = newpop

    print("GA finished, best score:", best_score)
    return best

# ---------------------------
# Synthesize simple audio from sequence
# ---------------------------
def synthesize(ind, filename="raag_bhairav_best.wav"):
    # Build audio as sum of tones per note with chosen durations
    freqs = [semitone_to_freq(BHAIRAV[p]) for p,d in ind]
    durs = [DURATIONS[d] for p,d in ind]

    # normalize total beat sum to desired DURATION_SEC
    total_beats = sum(durs)
    scale = DURATION_SEC / total_beats
    durs = [dd * scale for dd in durs]

    audio = np.array([], dtype=np.float32)
    for f, dur in zip(freqs, durs):
        t = np.linspace(0, dur, int(SR*dur), False)
        # simple ADSR-like amplitude envelope
        env = np.minimum(1, 5*t/dur) * np.minimum(1, (1 - (t/dur))*5)
        tone = 0.3 * env * np.sin(2 * np.pi * f * t)
        audio = np.concatenate((audio, tone))

    # normalize and write to wav
    audio_norm = audio / np.max(np.abs(audio)) * 0.95
    wavfile.write(filename, SR, (audio_norm * 32767).astype(np.int16))
    print("WAV written:", filename)

# ---------------------------
# Run and save
# ---------------------------
if __name__ == "__main__":
    best_ind = run_ga()
    print("Best individual (pitch_idx, dur_idx):")
    print(best_ind)
    synthesize(best_ind, filename="raag_bhairav_best.wav")

