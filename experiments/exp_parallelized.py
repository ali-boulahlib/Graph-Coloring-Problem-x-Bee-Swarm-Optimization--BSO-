import os
import glob
import csv
import time
import argparse
import random
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Dict, Tuple, List

#from dfs_coloring import load_graph
from bso_op import BSOColoring

# --- Configuration ---
GRAPHS_DIR = 'data/benchmarks'
OUTPUT_CSV = 'results/bso_experiments.csv'

# Parameter grid (excluding max_iter)
BASE_PARAM_GRID = {
    'n_bees':   [10, 20, 50],
    'max_steps': [5, 10, 15],
    'n_chance': [1, 3, 5],
    'flip':     [3, 5, 7],
}
# Time budgets for successive halving
BUDGETS = sorted([20, 50, 100])  # max_iter values
SAMPLE_SIZE = 30  # number of configs to sample per graph
SEEDS = [42]
TIMEOUT = 120  # seconds per run
REDUCTION_FACTOR = 3  # keep top 1/r

def load_graph(path: str) -> nx.Graph:
    
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'p':
                n = int(parts[1])
                G.add_nodes_from(range(1, n + 1))
            elif parts[0] == 'e':
                u, v = map(int, parts[1:])
                G.add_edge(u, v)
    return G

def single_run(graph_path: str, params: Dict) -> Tuple:
    graph_name = os.path.basename(graph_path)
    G = load_graph(graph_path)

    solver = BSOColoring(
        G,
        k_max=None,
        n_bees=params['n_bees'],
        max_steps=params['max_steps'],
        n_chance=params['n_chance'],
        max_iter=params['max_iter'],
        flip=params['flip'],
        seed=params['seed']
    )
    start = time.time()
    S, fit = solver.run()
    runtime = time.time() - start
    colors = fit[1]
    conflicts = fit[2]
    fit = fit[0]



    return (
        graph_name,
        params['seed'],
        params['n_bees'],
        params['max_steps'],
        params['n_chance'],
        params['max_iter'],
        params['flip'],
        fit,
        conflicts,
        colors,
        f"{runtime:.4f}",
        S # solution
    )


def run_experiments(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    graph_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, '*.txt')))

    # Prepare CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'graph','seed','n_bees','max_steps','n_chance','max_iter','flip',
            'fitness','conflicts','colors','runtime','solution'
        ])

        # For each graph
        for graph_path in graph_files:
            # Build base configs (exclude max_iter)
            base_configs = []
            for seed in SEEDS:
                for nb in BASE_PARAM_GRID['n_bees']:
                    for ms in BASE_PARAM_GRID['max_steps']:
                        for nc in BASE_PARAM_GRID['n_chance']:
                            for fl in BASE_PARAM_GRID['flip']:
                                base_configs.append({
                                    'seed': seed,
                                    'n_bees': nb,
                                    'max_steps': ms,
                                    'n_chance': nc,
                                    'flip': fl
                                })
            # random sample
            sampled = random.sample(base_configs, k=min(SAMPLE_SIZE, len(base_configs)))

            # Successive halving over budgets
            configs = sampled.copy()
            for budget in BUDGETS:
                # assign budget
                tier_params = []
                for p in configs:
                    p2 = p.copy()
                    p2['max_iter'] = budget
                    tier_params.append(p2)

                # run in parallel with timeout
                results: List[Tuple] = []
                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(single_run, graph_path, p): p for p in tier_params}
                    for fut in as_completed(futures):
                        p = futures[fut]
                        try:
                            res = fut.result(timeout=TIMEOUT)
                            print(f"✅ Completed: {graph_path}, {p}")
                            print(f"  Result: {res}")
                            results.append(res)
                            writer.writerow(res)
                        except TimeoutError:
                            print(f"⚠️ Timeout: {graph_path}, {p}")
                        except Exception as e:
                            print(f"❌  Error on {graph_path}, {p}: {e}")

                # select top 1/REDUCTION_FACTOR by fitness ascending
                results.sort(key=lambda r: int(r[7]))  # fitness at index 7
                k = max(1, len(results) // REDUCTION_FACTOR)
                # get configs for next round
                survivors = []
                for res in results[:k]:
                    # rebuild param dict
                    survivors.append({
                        'seed': res[1],
                        'n_bees': res[2],
                        'max_steps': res[3],
                        'n_chance': res[4],
                        'flip': res[6]
                    })
                configs = survivors

    print(f"Experiments complete. Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output','-o',default=OUTPUT_CSV)
    args = parser.parse_args()
    run_experiments(args.output)
