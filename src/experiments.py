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

# Parameter grid matching new implementation
PARAM_GRID = {
    'n_bees': [20, 30, 50],        # Swarm size
    'max_steps': [10, 15, 20],     # Local search depth
    'n_chance': [2, 3, 5],         # Diversity threshold
    'flip': [3, 5, 7],             # Search pattern width
    'max_iter': [200, 500, 1000]   # Iteration limit
}

# Seeds for stochastic averaging
SEEDS = [42, 113, 265, 491, 753]  # 5 different seeds

# Metrics header
CSV_COLUMNS = [
    'graph', 'nodes', 'edges', 'seed',
    'n_bees', 'max_steps', 'n_chance', 'flip', 'max_iter',
    'fitness', 'conflicts', 'colors', 'runtime', 'iterations',
    'diverse_injections', 'taboo_used'
]

def run_experiments(output_path: str):
    graph_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, '*.txt')))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS)
        
        for graph_path in graph_files:
            graph_name = os.path.basename(graph_path)
            G = load_graph(graph_path)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            # Full factorial parameter exploration
            for params in itertools.product(*PARAM_GRID.values()):
                n_bees, max_steps, n_chance, flip, max_iter = params
                
                for seed in SEEDS:
                    # Initialize solver with current parameters
                    solver = BSOColoring(
                        G,
                        k_max=n_nodes,
                        n_bees=n_bees,
                        max_steps=max_steps,
                        n_chance=n_chance,
                        flip=flip,
                        max_iter=max_iter,
                        seed=seed
                    )
                    
                    # Timed execution
                    start_time = time.time()
                    best_coloring, fitness = solver.run()
                    runtime = time.time() - start_time
                    
                    # Extract metrics
                    conflicts = fitness // solver.alpha
                    colors = fitness % solver.alpha
                    
                    writer.writerow([
                        graph_name, n_nodes, n_edges, seed,
                        n_bees, max_steps, n_chance, flip, max_iter,
                        fitness, conflicts, colors,
                        f"{runtime:.4f}", solver.iteration,
                        solver.diverse_injections,  # Track diversity events
                        len(solver.taboo_list)      # Memory usage
                    ])
                    
                    print(f"[{graph_name}] Bees: {n_bees} Steps: {max_steps} "
                          f"Flip: {flip} | Colors: {colors} Conflicts: {conflicts} "
                          f"Time: {runtime:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BSO Parameter Tuning')
    parser.add_argument('-o', '--output', required=True,
                        help='Output CSV path')
    args = parser.parse_args()
    
    run_experiments(args.output)