#!/usr/bin/env python3
"""
Reconstruct scrambled Lena using Genetic Algorithm (GA)
Assumes 512x512 image split into 16 tiles (4x4 grid, each 128x128).
Input: scrambled_lena.mat (Octave ASCII format)
Output: reconstructed_ga.png

Usage:
    python3 l4.py scrambled_lena.mat
"""

import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from time import time

# ---------- Step 1: Read ASCII Octave .mat ----------
def read_ascii_mat_matrix(path):
    # ... (this function is correct, no changes) ...
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != '']
    dims, start_idx = None, None
    for i, ln in enumerate(lines):
        if ln.startswith('#'): continue
        parts = ln.split()
        if len(parts) == 2:
            try:
                r, c = int(parts[0]), int(parts[1])
                dims = (r, c)
                start_idx = i + 1
                break
            except: pass
    if dims is None: raise ValueError("Matrix dimensions not found in file.")
    rows, cols = dims
    nums = []
    for ln in lines[start_idx:]:
        if ln.startswith('#'): continue
        for tok in ln.split():
            try: nums.append(int(tok))
            except: pass
    nums = nums[:rows * cols]
    arr = np.array(nums, dtype=np.uint8).reshape(rows, cols)
    return arr


# ---------- Step 2: Tile creation and cost ----------
def split_into_tiles(img, tile_size=128):
    # ... (this function is correct, no changes) ...
    rows, cols = img.shape
    nR, nC = rows // tile_size, cols // tile_size
    tiles = []
    for r in range(nR):
        for c in range(nC):
            tiles.append(img[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size])
    return tiles, nR, nC

def reassemble(perm, tiles, nR, nC, tile_size):
    # ... (this function is correct, no changes) ...
    out = np.zeros((nR*tile_size, nC*tile_size), dtype=np.uint8)
    for dest_idx, src_idx in enumerate(perm):
        r, c = dest_idx // nC, dest_idx % nC
        out[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tiles[src_idx]
    return out

def edge_cost(tile_a, tile_b, direction):
    # ... (this function is correct, no changes) ...
    if direction == 'right':
        diff = tile_a[:, -1].astype(np.int32) - tile_b[:, 0].astype(np.int32)
    elif direction == 'down':
        diff = tile_a[-1, :].astype(np.int32) - tile_b[0, :].astype(np.int32)
    return np.sum(diff * diff)

def total_cost(perm, tiles, nR, nC, tile_size):
    # ... (this function is correct, no changes) ...
    cost = 0
    for r in range(nR):
        for c in range(nC):
            idx = perm[r*nC + c]
            if c < nC - 1:
                right_idx = perm[r*nC + (c + 1)]
                cost += edge_cost(tiles[idx], tiles[right_idx], 'right')
            if r < nR - 1:
                down_idx = perm[(r + 1)*nC + c]
                cost += edge_cost(tiles[idx], tiles[down_idx], 'down')
    return cost


# ---------- Step 3: Genetic Algorithm (WITH ELITISM FIX) ----------
def genetic_algorithm(tiles, nR, nC, tile_size, pop_size=100, generations=1000, mutation_rate=0.2, elite_size=2):
    """Run GA to find best tile permutation."""
    n_tiles = len(tiles)
    
    def random_perm():
        p = list(range(n_tiles))
        random.shuffle(p)
        return p

    def crossover(p1, p2):
        a, b = sorted(random.sample(range(n_tiles), 2))
        child = [None]*n_tiles
        # Copy slice from parent 1
        child[a:b] = p1[a:b]
        # Fill the rest with genes from parent 2
        p2_genes = [gene for gene in p2 if gene not in child]
        child_idx = 0
        for i in range(n_tiles):
            if child[i] is None:
                child[i] = p2_genes[child_idx]
                child_idx += 1
        return child

    def mutate(p):
        i, j = random.sample(range(n_tiles), 2)
        p[i], p[j] = p[j], p[i]
        return p

    # Initialize population
    population = [random_perm() for _ in range(pop_size)]
    best_perm = None
    best_cost = float('inf')
    
    print("Starting Genetic Algorithm...")
    for gen in range(generations):
        # Calculate fitness for the entire population
        costs = [total_cost(p, tiles, nR, nC, tile_size) for p in population]
        
        # Find the best solution in the current generation
        current_best_idx = np.argmin(costs)
        current_best_cost = costs[current_best_idx]
        
        # Update the overall best solution found so far
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_perm = population[current_best_idx]

        # ###################### ELITISM FIX START ######################
        
        # Create the next generation
        new_pop = []
        
        # 1. Add the elite individuals (best from this generation) directly
        ranked_pop = [p for _, p in sorted(zip(costs, population))]
        for i in range(elite_size):
            new_pop.append(ranked_pop[i])

        # 2. Create the rest of the population through breeding
        while len(new_pop) < pop_size:
            # Tournament Selection
            p1_idx, p2_idx = random.sample(range(pop_size), 2)
            parent1 = population[p1_idx] if costs[p1_idx] < costs[p2_idx] else population[p2_idx]
            
            p3_idx, p4_idx = random.sample(range(pop_size), 2)
            parent2 = population[p3_idx] if costs[p3_idx] < costs[p4_idx] else population[p4_idx]
            
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            new_pop.append(child)
            
        population = new_pop
        
        # ###################### ELITISM FIX END ########################

        if gen % 50 == 0 or gen == generations - 1:
            print(f"Gen {gen:04d}: Best cost so far = {best_cost}")

    return best_perm, best_cost


# ---------- Step 4: Main ----------
def main():
    # ... (this section is correct, no changes) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', help='Path to scrambled_lena.mat')
    parser.add_argument('--tile-size', type=int, default=128)
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--pop-size', type=int, default=100)
    parser.add_argument('--mutation', type=float, default=0.08)
    parser.add_argument('--save-out', default='reconstructed_ga.png')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    start = time()
    print("Reading scrambled matrix...")
    img = read_ascii_mat_matrix(args.matfile)
    print("Shape:", img.shape)
    tiles, nR, nC = split_into_tiles(img, args.tile_size)
    print(f"Tiles: {nR} x {nC} = {len(tiles)}")
    best_perm, best_cost = genetic_algorithm(
        tiles, nR, nC, args.tile_size,
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation
    )
    print(f"GA complete. Best cost = {best_cost}")
    recon = reassemble(best_perm, tiles, nR, nC, args.tile_size)
    Image.fromarray(recon).save(args.save_out)
    print(f"Saved reconstructed image to {args.save_out}")
    if args.show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.title("Scrambled Input"); plt.imshow(img, cmap='gray'); plt.axis('off')
        plt.subplot(1, 2, 2); plt.title("Reconstructed (GA)"); plt.imshow(recon, cmap='gray'); plt.axis('off')
        plt.show()
    print(f"Total runtime: {time() - start:.1f} sec")

if __name__ == "__main__":
    main()
