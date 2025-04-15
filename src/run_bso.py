# src/run_bso.py

import argparse
import networkx as nx
from bso_coloring import BSOColoring

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to DIMACS .txt")
    p.add_argument("--k_max", type=int, default=None, help="Upper bound on colors")
    p.add_argument("--bees", type=int, default=30)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--chance", type=int, default=3)
    p.add_argument("--iter", type=int, default=500)
    return p.parse_args()

def load_graph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "p":
                n = int(parts[1])
                G.add_nodes_from(range(1, n+1))
            elif parts[0] == "e":
                u,v = map(int, parts[1:])
                G.add_edge(u, v)
    return G

def main():
    args = parse_args()
    G = load_graph(args.input)
    k_max = args.k_max or G.number_of_nodes()

    solver = BSOColoring(
        G,
        k_max=k_max,
        n_bees=args.bees,
        n_neighbors=args.neighbors,
        n_chance=args.chance,
        max_iter=args.iter
    )
    best, score = solver.run()
    conflicts = score // solver.alpha
    colors_used = score % solver.alpha

    print(f"Best fitness: {score} ({conflicts} conflicts, {colors_used} colors)")
    print(f"Chromatic estimate: {colors_used}")
    # Optionally save best to file

if __name__ == "__main__":
    main()
