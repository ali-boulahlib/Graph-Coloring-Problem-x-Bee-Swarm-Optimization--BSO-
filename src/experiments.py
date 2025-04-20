import os
import glob
import csv
import time
import random
import argparse

from bso_coloring import BSOColoring
from dfs_coloring import load_graph

# --- Configuration ---
GRAPHS_DIR = 'data/benchmarks'
# automatically include all .txt in benchmarks

# Parameter grid
PARAM_GRID = {
    'n_bees': [10, 20, 50],
    'n_neighbors': [5, 10, 20],
    'n_chance': [1, 3, 5],
    'max_iter': [10,20,50,100],
}
# seeds per setting for averaging
SEEDS = [42]

# metrics to record: graph, seed, parameters, fitness, conflicts, colors, runtime


def run_experiments(output_path: str):
    
    graph_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, '*.txt')))

    # prepare CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'graph', 'seed', 'n_bees', 'n_neighbors', 'n_chance', 'max_iter',
            'fitness', 'conflicts', 'colors', 'runtime'
        ])

        # iterate combinations
        for graph_path in graph_files:
            graph_name = os.path.basename(graph_path)
            G = load_graph(graph_path)
            for n_bees in PARAM_GRID['n_bees']:
                for n_neighbors in PARAM_GRID['n_neighbors']:
                    for n_chance in PARAM_GRID['n_chance']:
                        for max_iter in PARAM_GRID['max_iter']:
                            for seed in SEEDS:
                                random.seed(seed)
                                solver = BSOColoring(
                                    G,
                                    k_max=G.number_of_nodes(),
                                    n_bees=n_bees,
                                    n_neighbors=n_neighbors,
                                    n_chance=n_chance,
                                    max_iter=max_iter,
                                    seed=seed
                                )
                                start = time.time()
                                best_coloring, fitness = solver.run()
                                runtime = time.time() - start
                                # decode fitness
                                alpha = solver.alpha
                                conflicts = fitness // alpha
                                colors = fitness % alpha

                                writer.writerow([
                                    graph_name, seed, n_bees, n_neighbors, n_chance, max_iter,
                                    fitness, conflicts, colors, f"{runtime:.4f}"
                                ])
                                print(f"Done: {graph_name}, bees={n_bees}, neigh={n_neighbors}, "
                                      f"chance={n_chance}, iter={max_iter}, seed={seed}, "
                                      f"fitness={fitness}, time={runtime:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BSO parameter grid experiments')
    parser.add_argument('--output', '-o', default='results/bso_experiments.csv',
                        help='Path to CSV results file')
    args = parser.parse_args()
    run_experiments(args.output)
    print(f"Experiments complete. Results saved to {args.output}")
