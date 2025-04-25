import random
import numpy as np
import networkx as nx
from typing import List, Tuple
from collections import Counter

from utils import calculate_upper_bound, calculate_lower_bound

class BSOColoring:
    def __init__(
        self,
        G: nx.Graph,
        k_max: int,
        n_bees: int = 30,
        n_chance: int = 3,
        max_iter: int = 500,
        max_steps: int = 15,
        flip: int = 5,
        alpha: int = None,
        seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.G = G
        self.edges = list(G.edges())
        self.adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        self.n = G.number_of_nodes()
        self.k_max = calculate_upper_bound(G) if k_max is None else k_max
        self.n_bees = n_bees
        self.n_chance = n_chance
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.flip = flip
        self.alpha = alpha or (len(self.edges) + 1)

        # Initialize taboo list and reference solution
        self.taboo_list: List[np.ndarray] = []
        self.S_ref: np.ndarray = self.random_coloring()

    def random_coloring(self) -> np.ndarray:
        # Uniformly random colors in [1, k_max]
        return np.random.randint(1, self.k_max + 1, size=self.n)

    def fitness(self, S: np.ndarray) -> int:
        # Compute conflicts over cached edges
        conflicts = sum(1 for u, v in self.edges if S[u-1] == S[v-1])
        distinct = len(np.unique(S))
        return self.alpha * conflicts + distinct , distinct, conflicts

    def determine_search_area(self) -> List[np.ndarray]:
        search_area: List[np.ndarray] = []
        h = 0
        while len(search_area) < self.n_bees and h < self.flip:
            s = self.S_ref.copy()
            p = 0
            while self.flip * p + h < self.n:
                idx = self.flip * p + h
                current = int(s[idx])
                choices = list(range(1, self.k_max + 1))
                choices.remove(current)
                s[idx] = random.choice(choices)
                p += 1
            search_area.append(s)
            h += 1
        return search_area

    def hamming(self, A: np.ndarray, B: np.ndarray) -> int:
        return int(np.count_nonzero(A != B))

    def diversity(self, solution: np.ndarray) -> int:
        if not self.taboo_list:
            return 0
        return min(self.hamming(solution, taboo) for taboo in self.taboo_list)

    def inject_diversity(self, table: List[np.ndarray]) -> None:
        cnt = Counter(tuple(s.tolist()) for s in table)
        for sol_tuple, occ in cnt.items():
            if occ > self.n_chance:
                positions = [i for i, s in enumerate(table) if tuple(s.tolist()) == sol_tuple]
                for pos in positions[self.n_chance:]:
                    diversest = max(table, key=lambda T: sum(self.hamming(T, U) for U in table))
                    table[pos] = diversest.copy()

    def local_search(self, solution: List[int]) -> List[int]:
        # Perform local search from the given solution
        best_solution = solution.copy()
        best_fitness = self.fitness(solution)
        
        for _ in range(self.max_steps):
            # Find conflicting edges in the current solution
            conflicting_edges = [(u, v) for u, v in self.G.edges() if solution[u-1] == solution[v-1]]
            if not conflicting_edges:
                break  # No conflicts, exit early
            
            # Collect all nodes involved in conflicts (graph nodes, 1-based)
            conflicting_nodes = set()
            for u, v in conflicting_edges:
                conflicting_nodes.add(u)
                conflicting_nodes.add(v)
            conflicting_nodes = list(conflicting_nodes)
            
            # Randomly select a conflicting node
            node = random.choice(conflicting_nodes)
            
            # Determine current color (convert to 0-based index)
            current_color = solution[node - 1]
            
            # Generate possible new colors different from current
            available_colors = [c for c in range(1, self.k_max + 1) if c != current_color]
            if not available_colors:
                continue  # Skip if no colors available (unlikely case)
            new_color = random.choice(available_colors)
            
            # Create new solution by changing the node's color
            new_solution = solution.copy()
            new_solution[node - 1] = new_color
            
            # Calculate new fitness
            new_fitness = self.fitness(new_solution)
            
            # Update best solution if improved
            if new_fitness < best_fitness:
                best_solution = new_solution.copy()
                best_fitness = new_fitness
                solution = new_solution.copy()
        
        return best_solution

    def select_new_reference(
        self,
        dance_table: List[np.ndarray],
        chances: int
    ) -> Tuple[np.ndarray, int]:
        best_quality = min(dance_table, key=lambda s: self.fitness(s)[0])
        best_q_fit = self.fitness(best_quality)[0]
        current_fit = self.fitness(self.S_ref)[0]
        if best_q_fit < current_fit:
            return best_quality, self.n_chance
        else:
            chances -= 1
            if chances > 0:
                return best_quality, chances
            else:
                best_div = max(dance_table, key=lambda s: self.diversity(s))
                return best_div, self.n_chance

    def run(self) -> Tuple[np.ndarray, int]:
        chances = self.n_chance
        for _ in range(self.max_iter):
            self.taboo_list.append(self.S_ref.copy())
            search_area = self.determine_search_area()
            while len(search_area) < self.n_bees:
                s = self.S_ref.copy()
                idx = random.randrange(self.n)
                s[idx] = random.randint(1, self.k_max)
                search_area.append(s)

            dance_table = [self.local_search(s) for s in search_area]
            self.inject_diversity(dance_table)
            self.S_ref, chances = self.select_new_reference(dance_table, chances)

            if self.fitness(self.S_ref)[0] < self.alpha:
                break
        return self.S_ref, self.fitness(self.S_ref)
