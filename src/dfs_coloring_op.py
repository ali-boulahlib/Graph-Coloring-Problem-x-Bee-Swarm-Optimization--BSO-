#!/usr/bin/env python3
import time
import csv
import networkx as nx
import argparse
import os
import json
from typing import List, Tuple, Dict, Set

class DFSColoring:
    def __init__(
        self,
        input_path: str,
        log_path: str,
        timeout: float = 60.0,
    ):
        # Paths and timeout for each attempt
        self.input_path = input_path
        self.log_path = log_path
        self.timeout = timeout

        # Load and preprocess graph
        self.G = self.load_graph(self.input_path)
        # Compute deterministic bounds
        self.lb, self.ub = self.compute_bounds(self.G)

    def load_graph(self, path: str) -> nx.Graph:
        # Load graph from DIMACS file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file {path} does not exist")

        G = nx.Graph()
        node_count = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                parts = line.split()
                if parts[0] == "p":
                    if len(parts) >= 4 and parts[1] == "edge":
                        node_count = int(parts[2])
                        G.add_nodes_from(range(1, node_count + 1))
                    elif len(parts) >= 3:
                        node_count = int(parts[1])
                        G.add_nodes_from(range(1, node_count + 1))
                elif parts[0] == "e" and len(parts) >= 3:
                    u, v = int(parts[1]), int(parts[2])
                    if 1 <= u <= node_count and 1 <= v <= node_count:
                        G.add_edge(u, v)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def dsatur_coloring(self, G: nx.Graph) -> Tuple[int, Dict[int, int]]:
        # DSatur coloring algorithm
        coloring: Dict[int,int] = {}
        available_colors = {node: set(range(1, G.number_of_nodes() + 1)) for node in G.nodes()}
        uncolored = sorted(G.nodes(), key=lambda x: (-G.degree[x], x))

        first = uncolored.pop(0)
        coloring[first] = 1
        for nbr in G[first]:
            available_colors[nbr].discard(1)

        while uncolored:
            # Choose next by saturation degree, break ties by degree then id
            max_sat, next_node = -1, None
            for node in uncolored:
                used = {coloring.get(n) for n in G[node] if n in coloring}
                sat = len(used)
                if sat > max_sat or (sat == max_sat and next_node and (-G.degree[node], node) < (-G.degree[next_node], next_node)):
                    max_sat, next_node = sat, node
            color = min(available_colors[next_node])
            coloring[next_node] = color
            uncolored.remove(next_node)
            for nbr in G[next_node]:
                available_colors[nbr].discard(color)

        return max(coloring.values()), coloring

    def greedy_clique_size(self, G: nx.Graph) -> int:
        # Greedy algorithm to estimate clique number
        clique: List[int] = []
        candidates = set(G.nodes())
        while candidates:
            v = max(candidates, key=lambda x: (sum(1 for u in G[x] if u in candidates), -x))
            if all(u in G[v] for u in clique):
                clique.append(v)
            candidates.remove(v)
        return len(clique)

    def compute_bounds(self, G: nx.Graph) -> Tuple[int, int]:
        # Compute lower (clique) and upper (DSatur) bounds
        if not G.number_of_nodes():
            return 1, 1
        lb = self.greedy_clique_size(G)
        ub, _ = self.dsatur_coloring(G)
        return lb, ub

    def verify_coloring(self, G: nx.Graph, coloring: Dict[int,int]) -> bool:
        # Verify no adjacent nodes share the same color
        for u, v in G.edges():
            if u in coloring and v in coloring and coloring[u] == coloring[v]:
                return False
        return True

    def fitness(self, G: nx.Graph, S: List[int], alpha: float = 10.0, beta: float = 1.0) -> Tuple[float,int,int]:
        # Fitness: penalize conflicts, reward colored nodes, plus distinct count
        conflicts = sum(1 for u, v in G.edges() if S[u-1] == S[v-1])
        distinct = len({c for c in S if c != 0})
        uncolored = sum(1 for c in S if c == 0)
        val = alpha * conflicts - beta * uncolored + distinct
        return val, distinct, conflicts

    def is_k_colorable(self, G: nx.Graph, k: int) -> Tuple[bool, List[int], float]:
        # Classic DFS for k-coloring with timeout
        start = time.time()
        coloring: Dict[int,int] = {}
        best_solution = [0]*G.number_of_nodes()
        best_fit = -float('inf')
        nodes = sorted(G.nodes(), key=lambda x: (-G.degree[x], x))

        def dfs(idx: int) -> bool:
            nonlocal best_solution, best_fit
            # timeout
            if time.time() - start > self.timeout:
                S = [0]*G.number_of_nodes()
                for n,c in coloring.items(): S[n-1] = c
                f,_,_ = self.fitness(G,S)
                if f > best_fit:
                    best_fit, best_solution = f, S.copy()
                return False
            if idx >= len(nodes):
                return True
            node = nodes[idx]
            for color in range(1, k+1):
                if all(color != coloring.get(nbr) for nbr in G[node] if nbr in coloring):
                    coloring[node] = color
                    if len(coloring) % 50 == 0:
                        S = [0]*G.number_of_nodes()
                        for n,c in coloring.items(): S[n-1] = c
                        f,_,_ = self.fitness(G,S)
                        if f > best_fit:
                            best_fit, best_solution = f, S.copy()
                    if dfs(idx+1):
                        return True
                    del coloring[node]
            return False

        success = dfs(0)
        S_final = [0]*G.number_of_nodes()
        for n,c in coloring.items(): S_final[n-1] = c
        f_final,_,_ = self.fitness(G,S_final)
        if success:
            return True, S_final, f_final
        if f_final > best_fit:
            return False, S_final, f_final
        return False, best_solution, best_fit

    def find_chromatic_number(self) -> Tuple[int, List[int]]:
        # Binary search for chromatic number, logging each test
        low, high = self.lb, self.ub
        result = high
        final_sol: List[int] = []
        print(f"Starting binary search between {low} and {high}")

        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['k','colorable','time_seconds','fitness','solution'])
            while low <= high:
                mid = (low + high)//2
                print(f"Testing k={mid}")
                t0 = time.time()
                ok, sol, fit = self.is_k_colorable(self.G, mid)
                elapsed = time.time() - t0
                writer.writerow([mid, ok, elapsed, fit, json.dumps(sol)])
                csvfile.flush()
                print(f"k={mid}: {'Colorable' if ok else 'Not colorable'} in {elapsed:.2f}s, fitness: {fit:.2f}")
                if ok:
                    result, final_sol = mid, sol
                    high = mid - 1
                else:
                    low = mid + 1
        return result, final_sol

    def run(self):
        # Execute full pipeline
        print(f"Lower bound (Clique size): {self.lb}")
        print(f"Upper bound (DSatur): {self.ub}")
        chi, sol = self.find_chromatic_number()
        print(f"Chromatic number found: {chi}")

        coloring = {i+1: c for i,c in enumerate(sol) if c != 0}
        print(f"Sample coloring (first 10): {dict(list(coloring.items())[:10])}")
        valid = self.verify_coloring(self.G, coloring)
        print(f"Coloring is {'valid' if valid else 'invalid'}")

        # Save full solution array to text file
        out_file = os.path.splitext(self.log_path)[0] + '_coloring.txt'
        with open(out_file, 'w') as f:
            for node in range(1, self.G.number_of_nodes()+1):
                f.write(f"{node}: {sol[node-1]}\n")
        print(f"Full solution saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classic DFS Graph Coloring')
    parser.add_argument('input', help='Path to DIMACS graph file')
    parser.add_argument('log', help='Path to output CSV log')
    parser.add_argument('--timeout', type=float, default=60.0, help='Timeout per attempt')
    args = parser.parse_args()
    solver = DFSColoring(args.input, args.log, args.timeout)
    solver.run()
