import random
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import Counter

from utils import calculate_upper_bound, calculate_lower_bound

class BSOColoring:
    def __init__(
        self,
        G: nx.Graph,
        k_max: int = None,
        n_bees: int = 50,
        n_chance: int = 5,
        max_iter: int = 1000,
        max_steps: int = 30,
        flip: int = 5,
        alpha: int = None,
        tabu_tenure: int = 7,
        seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.G = G
        self.edges = list(G.edges())
        self.adj = {u: list(G.neighbors(u)) for u in G.nodes()}
        self.n = G.number_of_nodes()
        
        # Better k_max strategy
        lower_bound = calculate_lower_bound(G)
        upper_bound = calculate_upper_bound(G)
        #print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        self.k_max = k_max if k_max is not None else upper_bound
        
        # Ensure k_max is at least the upper bound of chromatic number
        if self.k_max < upper_bound:
            self.k_max = upper_bound
        
        self.n_bees = n_bees
        self.n_chance = n_chance
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.flip = flip
        self.alpha = alpha or (len(self.edges) * 3 + 1)  # Increased penalty for conflicts
        self.tabu_tenure = tabu_tenure

        # Initialize taboo list and reference solution
        self.taboo_list: List[np.ndarray] = []
        self.S_ref: np.ndarray = self.random_coloring()

    def random_coloring(self) -> np.ndarray:
        # Uniformly random colors in [1, k_max]
        return np.random.randint(1, self.k_max + 1, size=self.n)

    def greedy_coloring(self) -> np.ndarray:
        """Generate a solution using a greedy algorithm with randomized node order."""
        coloring = np.zeros(self.n, dtype=int)
        # Visit nodes in random order
        nodes = list(range(self.n))
        random.shuffle(nodes)
        
        for i in nodes:
            # Get colors of neighbors (1-based node indexing)
            neighbor_colors = {coloring[j-1] for j in self.adj[i+1] if coloring[j-1] > 0}
            
            # Find the lowest available color
            color = 1
            while color in neighbor_colors and color <= self.k_max:
                color += 1
            
            coloring[i] = color
        
        return coloring

    def fitness(self, S: np.ndarray) -> Tuple[int, int, int]:
        # Compute conflicts over cached edges
        conflicts = sum(1 for u, v in self.edges if S[u-1] == S[v-1])
        distinct = len(np.unique(S))
        return self.alpha * conflicts + distinct, distinct, conflicts

    def determine_search_area(self) -> List[np.ndarray]:
        search_area: List[np.ndarray] = []
        
        # Add the current best solution
        search_area.append(self.S_ref.copy())
        
        # Add traditional flip-based solutions
        h = 0
        while len(search_area) < self.n_bees // 2 and h < self.flip:
            s = self.S_ref.copy()
            p = 0
            while self.flip * p + h < self.n:
                idx = self.flip * p + h
                current = int(s[idx])
                choices = list(range(1, self.k_max + 1))
                if current in choices:
                    choices.remove(current)
                s[idx] = random.choice(choices)
                p += 1
            search_area.append(s)
            h += 1
        
        # Add completely random solutions for diversification
        while len(search_area) < 2 * self.n_bees // 3:
            search_area.append(self.random_coloring())
        
        # Add greedy solutions for intelligent starting points
        while len(search_area) < self.n_bees:
            search_area.append(self.greedy_coloring())
        
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
                    # Replace with a completely new solution rather than just the most diverse
                    if random.random() < 0.5:
                        table[pos] = self.random_coloring()
                    else:
                        table[pos] = self.greedy_coloring()

    def local_search(self, solution: np.ndarray) -> np.ndarray:
        best_solution = solution.copy()
        best_fitness, best_distinct, best_conflicts = self.fitness(solution)
        
        # Initialize tabu list for local search
        tabu_moves = {}  # (node, color) pairs that are forbidden
        
        for step in range(self.max_steps):
            # Exit early if we've found a solution with no conflicts
            if best_conflicts == 0:
                break
                
            # Find nodes involved in conflicts
            conflict_nodes = set()
            for u, v in self.edges:
                if solution[u-1] == solution[v-1]:
                    conflict_nodes.add(u-1)  # Convert to 0-based
                    conflict_nodes.add(v-1)
            
            if not conflict_nodes:
                break
                
            # Try to resolve conflicts by examining each conflict node
            improved = False
            # Convert to list and shuffle to avoid bias
            conflict_nodes_list = list(conflict_nodes)
            random.shuffle(conflict_nodes_list)
            
            for node in conflict_nodes_list:
                # Find colors that don't create conflicts with neighbors
                current_color = solution[node]
                neighbor_colors = {solution[neighbor-1] for neighbor in self.adj[node+1]}
                
                # Try colors that don't conflict with neighbors
                available_colors = [c for c in range(1, self.k_max + 1) if c not in neighbor_colors]
                random.shuffle(available_colors)  # Randomize color selection
                
                for color in available_colors:
                    # Skip tabu moves unless they lead to a new best solution (aspiration criterion)
                    if (node, color) in tabu_moves and step < tabu_moves[(node, color)]:
                        continue
                        
                    # Create new solution with this color change
                    new_solution = solution.copy()
                    new_solution[node] = color
                    new_fitness, new_distinct, new_conflicts = self.fitness(new_solution)
                    
                    if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_distinct < best_distinct):
                        best_solution = new_solution.copy()
                        best_fitness, best_distinct, best_conflicts = new_fitness, new_distinct, new_conflicts
                        # Add move to tabu list
                        tabu_moves[(node, color)] = step + self.tabu_tenure
                        improved = True
                        solution = new_solution.copy()  # Update current solution
                        break
                
                if improved:
                    break
                    
            # If no improvement was possible with non-conflicting colors, try any color change
            if not improved:
                node = random.choice(list(conflict_nodes))
                current_color = solution[node]
                available_colors = [c for c in range(1, self.k_max + 1) if c != current_color]
                
                if available_colors:
                    # Try each available color and pick the best one
                    best_candidate = None
                    best_candidate_fitness = float('inf')
                    best_candidate_color = None
                    
                    for color in available_colors:
                        # Skip tabu moves unless they lead to a new best solution
                        if (node, color) in tabu_moves and step < tabu_moves[(node, color)]:
                            continue
                            
                        candidate = solution.copy()
                        candidate[node] = color
                        candidate_fitness, _, candidate_conflicts = self.fitness(candidate)
                        
                        if candidate_fitness < best_candidate_fitness:
                            best_candidate = candidate
                            best_candidate_fitness = candidate_fitness
                            best_candidate_color = color
                    
                    if best_candidate is not None and best_candidate_fitness < best_fitness:
                        solution = best_candidate.copy()
                        best_solution = best_candidate.copy()
                        best_fitness = best_candidate_fitness
                        # Add move to tabu list
                        tabu_moves[(node, best_candidate_color)] = step + self.tabu_tenure
                    
            # If still no improvement, perform a random move to escape local optima
            if not improved and random.random() < 0.1:  # 10% chance
                node = random.choice(range(self.n))
                current_color = solution[node]
                new_color = random.choice([c for c in range(1, self.k_max + 1) if c != current_color])
                solution[node] = new_color
                
        return best_solution

    def select_new_reference(
        self,
        dance_table: List[np.ndarray],
        chances: int
    ) -> Tuple[np.ndarray, int]:
        # Find solutions with minimum conflicts
        min_conflicts = float('inf')
        min_conflict_solutions = []
        
        for solution in dance_table:
            _, _, conflicts = self.fitness(solution)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                min_conflict_solutions = [solution]
            elif conflicts == min_conflicts:
                min_conflict_solutions.append(solution)
        
        # If we found solutions with fewer conflicts than current best
        current_fitness, _, current_conflicts = self.fitness(self.S_ref)
        
        if min_conflicts < current_conflicts:
            # Choose the solution with the minimum number of colors among those with min conflicts
            best_solution = min(min_conflict_solutions, key=lambda s: self.fitness(s)[1])
            return best_solution, self.n_chance
        else:
            # If no improvement, reduce chances
            chances -= 1
            if chances > 0:
                # Return current best from dance table
                best_quality = min(dance_table, key=lambda s: self.fitness(s)[0])
                return best_quality, chances
            else:
                # Reset chances and select most diverse solution
                best_div = max(dance_table, key=lambda s: self.diversity(s))
                return best_div, self.n_chance

    def run(self, max_restarts=5) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        overall_best = None
        overall_best_fitness = float('inf')
        overall_best_conflicts = float('inf')
        
        for restart in range(max_restarts):
            # Reset for this restart
            self.taboo_list = []
            self.S_ref = self.greedy_coloring()  # Start with greedy solution
            chances = self.n_chance
            no_improvement_counter = 0
            
            for iteration in range(self.max_iter):
                # Add current reference to taboo list
                if len(self.taboo_list) >= 50:  # Limit taboo list size to prevent memory issues
                    self.taboo_list.pop(0)
                self.taboo_list.append(self.S_ref.copy())
                
                # Generate search area
                search_area = self.determine_search_area()
                
                # Perform local search for each solution in the search area
                dance_table = [self.local_search(s) for s in search_area]
                
                # Inject diversity into dance table
                self.inject_diversity(dance_table)
                
                # Select new reference solution
                old_fitness = self.fitness(self.S_ref)[0]
                self.S_ref, chances = self.select_new_reference(dance_table, chances)
                new_fitness = self.fitness(self.S_ref)[0]
                
                # Track improvement
                if new_fitness >= old_fitness:
                    no_improvement_counter += 1
                else:
                    no_improvement_counter = 0
                
                # Break early if we've found a valid coloring with minimum colors
                fitness, distinct, conflicts = self.fitness(self.S_ref)
                if conflicts == 0:
                    print(f"⭐ Restart {restart+1}, Iteration {iteration+1}: Found valid coloring with {distinct} colors")
                    
                    # If this is better than our overall best solution
                    if fitness < overall_best_fitness:
                        overall_best = self.S_ref.copy()
                        overall_best_fitness = fitness
                        overall_best_conflicts = conflicts
                    
                    # If we've reached the lower bound, we can stop
                    lower_bound = calculate_lower_bound(self.G)
                    if distinct <= lower_bound:
                        print(f"🎉 Optimal solution found! Using {distinct} colors (lower bound: {lower_bound})")
                        return self.S_ref, self.fitness(self.S_ref)
                    
                    break
                
                # Update overall best if needed
                if fitness < overall_best_fitness:
                    overall_best = self.S_ref.copy()
                    overall_best_fitness = fitness
                    overall_best_conflicts = conflicts
                
                # Print progress every 100 iterations
                if iteration % 100 == 0:
                    print(f"Restart {restart+1}, Iteration {iteration}: Best fitness = {fitness}, Conflicts = {conflicts}, Colors = {distinct}")
                
                # Break early if no improvement for a while
                if no_improvement_counter >= 50:
                    print(f"No improvement for 50 iterations, breaking out of this restart")
                    break
            
            print(f"Restart {restart+1}/{max_restarts} complete. Best fitness: {overall_best_fitness}, Conflicts: {overall_best_conflicts}")
            
            # If we found a valid coloring and reached restart limit
            if overall_best_conflicts == 0 and restart >= max_restarts - 1:
                print("Found a valid coloring, stopping restarts")
                break
        
        # Return the overall best solution
        if overall_best is None:
            return self.S_ref, self.fitness(self.S_ref)
        else:
            return overall_best, self.fitness(overall_best)

    def optimize_num_colors(self, max_attempts=3) -> Tuple[np.ndarray, int]:
        """
        Try to find a valid coloring with progressively fewer colors.
        """
        # First find any valid coloring
        solution, (_, distinct, conflicts) = self.run(max_restarts=2)
        
        if conflicts > 0:
            print(f"Could not find a valid coloring initially. Conflicts: {conflicts}")
            return solution, (distinct, conflicts)
        
        print(f"Starting with a valid {distinct}-coloring")
        best_solution = solution.copy()
        best_distinct = distinct
        
        # Try to reduce the number of colors
        for attempt in range(max_attempts):
            target_colors = best_distinct - 1
            if target_colors < calculate_lower_bound(self.G):
                print(f"Reached lower bound of {calculate_lower_bound(self.G)} colors")
                break
                
            print(f"Attempting to find a {target_colors}-coloring")
            self.k_max = target_colors
            
            # Reset and try again with fewer colors
            temp_solution, (_, new_distinct, new_conflicts) = self.run(max_restarts=2)
            
            if new_conflicts == 0:
                print(f"Success! Found a valid {new_distinct}-coloring")
                best_solution = temp_solution.copy()
                best_distinct = new_distinct
            else:
                print(f"Failed to find a valid {target_colors}-coloring")
                break
        
        return best_solution, best_distinct