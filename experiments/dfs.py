import time
import csv
import networkx as nx
import argparse
import os
import json
from typing import List, Tuple, Dict, Set


# Load graph from DIMACS file
def load_graph(path: str) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file {path} does not exist")

    G = nx.Graph()
    node_count = 0
    edge_count = 0

    try:
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
                        edge_count += 1

        print(
            f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    except Exception as e:
        raise ValueError(f"Error reading file {path}: {e}")


# DSatur coloring algorithm
def dsatur_coloring(G: nx.Graph) -> Tuple[int, Dict[int, int]]:
    coloring = {}
    available_colors = {
        node: set(range(1, G.number_of_nodes() + 1)) for node in G.nodes()
    }
    uncolored = sorted(G.nodes(), key=lambda x: (-G.degree[x], x))

    first_node = uncolored[0]
    coloring[first_node] = 1
    uncolored.remove(first_node)
    for neighbor in G[first_node]:
        if neighbor in available_colors:
            available_colors[neighbor].discard(1)

    while uncolored:
        max_sat = -1
        next_node = None
        for node in uncolored:
            used_colors = {coloring.get(n) for n in G[node] if n in coloring}
            sat_degree = len(used_colors)
            if sat_degree > max_sat or (
                sat_degree == max_sat
                and (-G.degree[node], node) < (-G.degree[next_node], next_node)
                if next_node
                else True
            ):
                max_sat = sat_degree
                next_node = node

        color = min(available_colors[next_node])
        coloring[next_node] = color
        uncolored.remove(next_node)
        for neighbor in G[next_node]:
            if neighbor in available_colors:
                available_colors[neighbor].discard(color)

    return max(coloring.values()), coloring


# Greedy algorithm to estimate clique number (lower bound)
def greedy_clique_size(G: nx.Graph) -> int:
    clique = []
    candidates = set(G.nodes())

    while candidates:
        v = max(candidates, key=lambda x: (sum(1 for u in G[x] if u in candidates), -x))
        if all(u in G[v] for u in clique):
            clique.append(v)
        candidates.remove(v)

    return len(clique)


# Compute deterministic bounds
def compute_bounds(G: nx.Graph) -> Tuple[int, int]:
    if not G.number_of_nodes():
        return 1, 1
    lb = greedy_clique_size(G)
    ub, _ = dsatur_coloring(G)
    return lb, ub


# Verify coloring
def verify_coloring(G: nx.Graph, coloring: Dict[int, int]) -> bool:
    for u, v in G.edges():
        if u in coloring and v in coloring and coloring[u] == coloring[v]:
            return False
    return True


# Fitness function with uncolored nodes as conflicts
def fitness(
    G: nx.Graph, S: List[int], alpha: float = 10.0, beta: float = 1.0
) -> Tuple[float, int, int]:
    conflicts = sum(
        1 for u, v in G.edges() if S[u - 1] == S[v - 1]
    )  # Count same colors, including 0
    distinct = len(set(c for c in S if c != 0))  # Non-zero colors only
    uncolored = sum(1 for c in S if c == 0)
    fitness_val = alpha * conflicts - beta * uncolored + distinct
    return fitness_val, distinct, conflicts


# Classic DFS for k-coloring
def is_k_colorable(
    G: nx.Graph, k: int, timeout: float = 60
) -> Tuple[bool, List[int], float]:
    start_time = time.time()
    coloring = {}
    best_solution = [0] * G.number_of_nodes()  # Initialize full solution array
    best_fitness = -float("inf")

    # Sort nodes by degree (highest first)
    nodes = sorted(G.nodes(), key=lambda x: (-G.degree[x], x))

    def dfs(index: int) -> bool:
        nonlocal best_solution, best_fitness

        # Check timeout
        if time.time() - start_time > timeout:
            # Convert coloring to full solution array
            S = [0] * G.number_of_nodes()
            for node, color in coloring.items():
                S[node - 1] = color
            current_fitness, _, _ = fitness(G, S)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = S.copy()
            return False

        # Base case: all nodes colored
        if index >= len(nodes):
            return True

        node = nodes[index]
        for color in range(1, k + 1):
            if all(color != coloring.get(nbr) for nbr in G[node] if nbr in coloring):
                coloring[node] = color
                # Save partial solution every 50 nodes
                if len(coloring) % 50 == 0:
                    S = [0] * G.number_of_nodes()
                    for n, c in coloring.items():
                        S[n - 1] = c
                    current_fitness, _, _ = fitness(G, S)
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_solution = S.copy()

                if dfs(index + 1):
                    return True
                del coloring[node]

        return False

    # Start DFS
    result = dfs(0)

    # Return best solution found
    S = [0] * G.number_of_nodes()
    for node, color in coloring.items():
        S[node - 1] = color
    fitness_val, distinct, conflicts = fitness(G, S)
    if result:
        return True, S, fitness_val
    else:
        # Return the best full solution array
        if fitness_val > best_fitness:
            return False, S, fitness_val
        return False, best_solution, best_fitness


# Binary search for chromatic number
def find_chromatic_number(
    G: nx.Graph, lb: int, ub: int, log_path: str
) -> Tuple[int, List[int]]:
    result = ub
    final_solution = None

    print(f"Starting binary search between {lb} and {ub}")

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["k", "colorable", "time_seconds", "fitness", "solution"])

        low, high = lb, ub
        while low <= high:
            mid = (low + high) // 2
            print(f"Testing k={mid}")

            start = time.time()
            colorable, solution, fitness = is_k_colorable(G, mid)
            elapsed = time.time() - start

            solution_json = json.dumps(solution)
            writer.writerow([mid, colorable, elapsed, fitness, solution_json])
            csvfile.flush()

            print(
                f"k={mid}: {'Colorable' if colorable else 'Not colorable'} in {elapsed:.2f}s, fitness: {fitness:.2f}"
            )

            if colorable:
                result = mid
                final_solution = solution
                high = mid - 1
            else:
                low = mid + 1

    return result, final_solution


# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Classic DFS Graph Coloring with Full Solution Array"
    )
    parser.add_argument("input", help="Path to DIMACS graph file")
    parser.add_argument("log", help="Path to output CSV log")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each coloring attempt",
    )

    args = parser.parse_args()

    try:
        start_total = time.time()

        G = load_graph(args.input)

        print("Calculating bounds...")
        lb, ub = compute_bounds(G)

        print(f"Lower bound (Clique size): {lb}")
        print(f"Upper bound (DSatur): {ub}")

        chi, solution = find_chromatic_number(G, lb, ub, args.log)

        elapsed = time.time() - start_total
        print(f"\nChromatic number found: {chi}")
        print(f"Total time: {elapsed:.2f} seconds")

        if solution:
            # Convert solution to dictionary for sampling
            coloring = {i + 1: c for i, c in enumerate(solution) if c != 0}
            print(f"Final coloring (sample): {dict(list(coloring.items())[:10])}")
            is_valid = verify_coloring(G, coloring)
            print(f"Coloring is {'valid' if is_valid else 'invalid'}")

            # Save full solution array
            coloring_file = f"{os.path.splitext(args.log)[0]}_coloring.txt"
            with open(coloring_file, "w") as f:
                for node in range(1, G.number_of_nodes() + 1):
                    color = solution[node - 1]
                    f.write(f"{node}: {color}\n")
            print(f"Full solution saved to {coloring_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
