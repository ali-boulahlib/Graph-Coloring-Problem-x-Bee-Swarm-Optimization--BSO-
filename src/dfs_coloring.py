import networkx as nx
import time
import argparse
from typing import Dict, Tuple, Optional

class DFSColoring:
    def __init__(self, G: nx.Graph, k_max: Optional[int] = None):

        self.G = G                                                                             # Graph to color
        self.k_max = k_max                                                                     # Max k to try
        self.n = G.number_of_nodes()                                                           # Number of nodes

        self.order = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True) # Order nodes by degree (highest first)
        self.k_max = k_max or self.n
        
        self.coloring: Dict[int, int] = {v: 0 for v in G.nodes()}
                                     # Initialize coloring (0 = uncolored)

    def valid(self, v: int, c: int) -> bool:
        # Check if color c can be assigned to node v without conflicts
        return all(self.coloring[u] != c for u in self.G.neighbors(v))

    def dfs(self, idx: int, k: int) -> bool:
        
        ## Recursive DFS to try coloring order[idx:] with k colors.
        if idx == len(self.order):
            print(f"Found solution with {k} colors: {self.coloring}")
            
            return True
        v = self.order[idx]
        for color in range(1, k+1):
            if self.valid(v, color):
                self.coloring[v] = color
                if self.dfs(idx + 1, k):
                    return True
                # backtrack
                self.coloring[v] = 0
        return False

    def run(self) -> Tuple[int, Dict[int, int], float]:
        """
        Try k from 1 to k_max. Returns:
        - best_k: smallest k for which a coloring was found
        - coloring: final node->color map
        - elapsed time in seconds
        """
        start = time.time()
        for k in range(1, self.k_max + 1):
            # reset coloring
            for v in self.G.nodes():
                self.coloring[v] = 0
            if self.dfs(0, k):
                elapsed = time.time() - start
                return k, self.coloring.copy(), elapsed
        elapsed = time.time() - start
        # if no solution found up to k_max, return k_max
        return self.k_max, self.coloring.copy(), elapsed


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFS-based graph coloring baseline')
    parser.add_argument('--input', '-i', required=True, help='Path to DIMACS graph file')
    parser.add_argument('--kmax', type=int, default=None, help='Max k to try (default = n)')
    args = parser.parse_args()

    G = load_graph(args.input)
    solver = DFSColoring(G, args.kmax)
    best_k, coloring, runtime = solver.run()
    print(f"Exact chromatic number: {best_k} (computed in {runtime:.3f}s)")
