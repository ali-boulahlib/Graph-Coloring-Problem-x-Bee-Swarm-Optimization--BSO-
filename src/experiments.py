import os
import glob
import csv
import time
import random
import argparse
import networkx as nx

from typing import List, Tuple

# Import the provided BSOColoring class
# Assuming the BSOColoring implementation is in the same file
from bso_coloring import BSOColoring
from dfs_coloring_op import load_graph


# --- Configuration ---
GRAPHS_DIR = 'data/benchmarks'
# automatically include all .txt in benchmarks

# Parameter grid adjusted for the provided implementation
PARAM_GRID = {
    'n_bees': [10, 20, 50],
    'max_steps': [5, 10, 15],  # replaces n_neighbors with max_steps
    'n_chance': [1, 3, 5],
    'max_iter': [20, 50, 100],
    'flip': [3, 5, 7]  # added the flip parameter
}
# seeds per setting for averaging
SEEDS = [42]





def run_experiments(output_path: str):
    
    graph_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, '*.txt')))

    # prepare CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'graph', 'seed', 'n_bees', 'max_steps', 'n_chance', 'max_iter', 'flip',
            'fitness', 'conflicts', 'colors', 'runtime'
        ])

        # iterate combinations
        for graph_path in graph_files:
            graph_name = os.path.basename(graph_path)
            G = load_graph(graph_path)
            for n_bees in PARAM_GRID['n_bees']:
                for max_steps in PARAM_GRID['max_steps']:
                    for n_chance in PARAM_GRID['n_chance']:
                        for max_iter in PARAM_GRID['max_iter']:
                            for flip in PARAM_GRID['flip']:
                                for seed in SEEDS:
                                    print(f"Running: {graph_name}, bees={n_bees}, steps={max_steps}, "
                                          f"chance={n_chance}, iter={max_iter}, flip={flip}, seed={seed}")
                                    
                                    solver = BSOColoring(
                                        G,
                                        k_max=None,
                                        n_bees=n_bees,
                                        max_steps=max_steps,  # using max_steps instead of n_neighbors
                                        n_chance=n_chance,
                                        max_iter=max_iter,
                                        flip=flip,  # added flip parameter
                                        seed=seed
                                    )
                                    
                                    start = time.time()
                                    best_coloring, fitness = solver.run()
                                    runtime = time.time() - start
                                    
                                    # decode fitness
                                    colors = fitness[1]
                                    conflicts = fitness[2]
                                    fitness = fitness[0]

                                    writer.writerow([
                                        graph_name, seed, n_bees, max_steps, n_chance, max_iter, flip,
                                        fitness, conflicts, colors, f"{runtime:.4f}"
                                    ])
                                    
                                    print(f"Done: fitness={fitness}, conflicts={conflicts}, "
                                          f"colors={colors}, time={runtime:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BSO parameter grid experiments')
    parser.add_argument('--output', '-o', default='results/bso_experiments.csv',
                        help='Path to CSV results file')
    args = parser.parse_args()
    run_experiments(args.output)
    print(f"Experiments complete. Results saved to {args.output}")