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
        # --- Initialization of parameters and data structures ---
        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Store the input graph and basic derived attributes
        self.G = G
        self.edges = list(G.edges())                # Cached list of edges for conflict checking
        self.adj = {u: list(G.neighbors(u))         # Adjacency list map node -> neighbors
                    for u in G.nodes()}
        self.n = G.number_of_nodes()                # Total number of vertices
        
        # Determine lower and upper bounds on the chromatic number
        lower_bound = calculate_lower_bound(G) 
        upper_bound = calculate_upper_bound(G)
        # If user didn't supply k_max, use the computed upper bound
        self.k_max = k_max if k_max is not None else upper_bound
        # Ensure k_max is at least the upper bound
        if self.k_max < upper_bound:
            self.k_max = upper_bound
        
        # BSO-specific parameters
        self.n_bees = n_bees            # Swarm size (# of candidate solutions)
        self.n_chance = n_chance        # Duplicate threshold before injecting diversity
        self.max_iter = max_iter        # Max BSO iterations (dance phases)
        self.max_steps = max_steps      # Max steps in inner tabu local search
        self.flip = flip                # Flip parameter for neighborhood generation
        # Alpha large -> heavy penalty on conflicts; default = 3*|E| + 1
        self.alpha = alpha or (len(self.edges) * 3 + 1)
        self.tabu_tenure = tabu_tenure  # How long a move stays tabu

        # Initialize history and reference solution
        self.taboo_list: List[np.ndarray] = []        # Stores recent reference solutions
        self.S_ref: np.ndarray = self.random_coloring()  # Current best (reference) coloring

    def random_coloring(self) -> np.ndarray:
        """
        Generate a random coloring: assign each node a color in [1..k_max] uniformly at random.
        Returns a length-n numpy array of integers.
        """
        return np.random.randint(1, self.k_max + 1, size=self.n)

    def greedy_coloring(self) -> np.ndarray:
        """
        Generate an initial coloring via a randomized greedy heuristic:
        1. Shuffle nodes in random order.
        2. For each node, pick the smallest color not used by its already-colored neighbors.
        """
        coloring = np.zeros(self.n, dtype=int)
        nodes = list(range(self.n))
        random.shuffle(nodes)
        for i in nodes:
            # Gather used neighbor colors (1-based node indexing -> adjust by -1)
            neighbor_colors = {
                coloring[j-1]
                for j in self.adj[i+1]
                if coloring[j-1] > 0
            }
            # Find lowest available color
            color = 1
            while color in neighbor_colors and color <= self.k_max:
                color += 1
            coloring[i] = color  # Might exceed k_max by 1 if neighbors used all
        return coloring

    def fitness(self, S: np.ndarray) -> Tuple[int, int, int]:
        """
        Compute the fitness of coloring S:
          - conflicts = number of monochromatic edges
          - distinct = number of distinct colors used
        Returns a tuple: (alpha * conflicts + distinct, distinct, conflicts)
        """
        conflicts = sum(1 for u, v in self.edges if S[u-1] == S[v-1])
        distinct = len(np.unique(S))
        return self.alpha * conflicts + distinct, distinct, conflicts

    def determine_search_area(self) -> List[np.ndarray]:
        """
        Build the "dance table" of candidate solutions around the reference solution:
          1. Keep the reference S_ref.
          2. Generate up to n_bees/2 flip-based neighbors (systematic small perturbations).
          3. Fill next ~1/6 of swarm with purely random colorings for diversification.
          4. Fill remaining slots with greedy colorings for more informed starts.
        """
        search_area: List[np.ndarray] = []
        # 1. Reference solution
        search_area.append(self.S_ref.copy())
        # 2. Flip-based neighbors
        h = 0
        while len(search_area) < self.n_bees // 2 and h < self.flip:
            s = self.S_ref.copy()
            p = 0
            # Flip every "flip"-th position starting at offset h
            while self.flip * p + h < self.n:
                idx = self.flip * p + h
                current = int(s[idx])
                # Build choice list excluding the current color
                choices = list(range(1, self.k_max + 1))
                if current in choices:
                    choices.remove(current)

                # Randomly reassign the new color
                s[idx] = random.choice(choices)
                p += 1
            search_area.append(s)
            h += 1


        # 3. Random diversifiers
        while len(search_area) < 2 * self.n_bees // 3:
            search_area.append(self.random_coloring())

        # 4. Greedy seeds
        while len(search_area) < self.n_bees:
            search_area.append(self.greedy_coloring())

        return search_area

    def hamming(self, A: np.ndarray, B: np.ndarray) -> int:
        return int(np.count_nonzero(A != B))

    def diversity(self, solution: np.ndarray) -> int:
        # return the smallest hamming distance to any solution in the taboo list
        if not self.taboo_list:
            return 0
        return min(self.hamming(solution, taboo) for taboo in self.taboo_list)

    def inject_diversity(self, table: List[np.ndarray]) -> None:
        """
        Detect duplicates in the candidate table: if a coloring occurs > n_chance times,
        replace the extras with brand-new (random or greedy) solutions.
        """
        cnt = Counter(tuple(s.tolist()) for s in table)
        for sol_tuple, occ in cnt.items():
            if occ > self.n_chance:
                positions = [i for i, s in enumerate(table) if tuple(s.tolist()) == sol_tuple]
                for pos in positions[self.n_chance:]:
                    # 50% chance to regenerate randomly, otherwise via greedy
                    if random.random() < 0.5:
                        table[pos] = self.random_coloring()
                    else:
                        table[pos] = self.greedy_coloring()

    def local_search(self, solution: np.ndarray) -> np.ndarray:
        
        best_solution = solution.copy()
        best_fitness, best_distinct, best_conflicts = self.fitness(solution)
        tabu_moves: Dict[Tuple[int,int], int] = {}

        for step in range(self.max_steps):


            # Early exit
            if best_conflicts == 0:
                break


            # Identify nodes involved in conflicts
            conflict_nodes: Set[int] = set()
            for u, v in self.edges:
                if solution[u-1] == solution[v-1]:
                    conflict_nodes.add(u-1)
                    conflict_nodes.add(v-1)
            if not conflict_nodes:
                break

            # I- Phase I: Conflict-Free Recoloring
            improved = False
            for node in random.sample(list(conflict_nodes), len(conflict_nodes)):
                current_color = solution[node]
                neighbor_colors = {solution[n-1] for n in self.adj[node+1]}
                # Try conflict-free colors first
                available_colors = [c for c in range(1, self.k_max+1) if c not in neighbor_colors]
                random.shuffle(available_colors)
                for color in available_colors:
                    # Skip tabu unless aspiration
                    if (node, color) in tabu_moves and step < tabu_moves[(node, color)]:
                        continue
                    # Evaluate move
                    new_solution = solution.copy()
                    new_solution[node] = color
                    new_fitness, new_distinct, new_conflicts = self.fitness(new_solution)
                    # Accept if fewer conflicts or same conflicts but fewer colors
                    if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and new_distinct < best_distinct):
                        best_solution = new_solution.copy()
                        best_fitness, best_distinct, best_conflicts = new_fitness, new_distinct, new_conflicts
                        tabu_moves[(node, color)] = step + self.tabu_tenure
                        solution = new_solution.copy()
                        improved = True
                        break
                if improved:
                    break

            # If no improvement from conflict-free colors, try any color change
            if not improved:
                node = random.choice(list(conflict_nodes))
                current_color = solution[node]
                candidates = [c for c in range(1, self.k_max+1) if c != current_color]
                if candidates:
                    best_candidate, best_candidate_fitness = None, float('inf')
                    best_candidate_color = None
                    for color in candidates:
                        if (node, color) in tabu_moves and step < tabu_moves[(node, color)]:
                            continue
                        cand = solution.copy()
                        cand[node] = color
                        f, _, _ = self.fitness(cand)
                        if f < best_candidate_fitness:
                            best_candidate, best_candidate_fitness, best_candidate_color = cand, f, color
                    if best_candidate_fitness < best_fitness:
                        best_solution = best_candidate.copy()
                        best_fitness = best_candidate_fitness
                        tabu_moves[(node, best_candidate_color)] = step + self.tabu_tenure
                        solution = best_candidate.copy()

            # Random move escape (10% chance)
            if not improved and random.random() < 0.1:
                node = random.randrange(self.n)
                current_color = solution[node]
                options = [c for c in range(1, self.k_max+1) if c != current_color]
                if options:
                    solution[node] = random.choice(options)

        return best_solution

    def select_new_reference(
        self,
        dance_table: List[np.ndarray],
        chances: int
    ) -> Tuple[np.ndarray, int]:
        """
        Choose the next reference solution from the completed dance table:
          1. Gather all with minimal conflicts.
          2. If they improve over current S_ref, pick the one with fewest colors and reset chances.
          3. Otherwise decrement chances: if still >0, pick best by overall fitness; if 0, pick most diverse and reset.
        Returns (new_ref, updated_chances).
        """
        min_conflicts = float('inf')
        min_sols: List[np.ndarray] = []
        for sol in dance_table:
            _, _, c = self.fitness(sol)
            if c < min_conflicts:
                min_conflicts, min_sols = c, [sol]
            elif c == min_conflicts:
                min_sols.append(sol)

        # Compare to current reference
        _, _, curr_conf = self.fitness(self.S_ref)
        if min_conflicts < curr_conf:
            # Better conflict count -> choose among those the fewest colors
            best = min(min_sols, key=lambda s: self.fitness(s)[1])
            return best, self.n_chance
        else:
            # No conflict improvement
            chances -= 1
            if chances > 0:
                # Still can try best quality
                best_q = min(dance_table, key=lambda s: self.fitness(s)[0])
                return best_q, chances
            else:
                # Force diversity
                best_d = max(dance_table, key=lambda s: self.diversity(s))
                return best_d, self.n_chance

    def run(self, max_restarts=5) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Main BSO driver with optional restarts:
          - For each restart: clear taboo_list, seed S_ref with greedy, run dance phases.
          - Track overall best across all restarts.
          - Early exit if theoretical lower bound reached.
        Returns (best_coloring, fitness_tuple).
        """
        overall_best, best_fit, best_conf = None, float('inf'), float('inf')
        for r in range(max_restarts):
            # Reset for this restart
            self.taboo_list.clear()
            self.S_ref = self.greedy_coloring()
            chances = self.n_chance
            no_improve = 0

            for it in range(self.max_iter):
                # Maintain a sliding-window taboo list of references
                if len(self.taboo_list) >= 50:
                    self.taboo_list.pop(0)
                self.taboo_list.append(self.S_ref.copy())

                # Generate neighborhood and improve each
                dance_area = self.determine_search_area()
                dance = []
                for s in dance_area:
                    dance.append(self.local_search(s))
                # Inject diversity among improved solutions
                self.inject_diversity(dance)
                # Select next S_ref and update chances
                old_f = self.fitness(self.S_ref)[0]
                self.S_ref, chances = self.select_new_reference(dance, chances)
                new_f = self.fitness(self.S_ref)[0]
                no_improve = no_improve + 1 if new_f >= old_f else 0

                fit, distinct, conf = self.fitness(self.S_ref)
                # Record if best overall
                if fit < best_fit:
                    overall_best, best_fit, best_conf = self.S_ref.copy(), fit, conf
                # Early break on zero conflicts
                if conf == 0:
                    lb = calculate_lower_bound(self.G)
                    if distinct <= lb:
                        return self.S_ref, self.fitness(self.S_ref)
                    break
                # Break if stuck
                if no_improve >= 50:
                    break
            # Continue to next restart
        # Return the best solution found
        if overall_best is None:
            return self.S_ref, self.fitness(self.S_ref)
        return overall_best, (best_fit // self.alpha, best_fit % self.alpha, best_conf)

    def optimize_num_colors(self, max_attempts=3) -> Tuple[np.ndarray, int]:
        """
        After finding any valid coloring, attempt to reduce the color count by lowering k_max
        and re-running the BSO up to max_attempts times.
        """
        sol, (_, distinct, conf) = self.run(max_restarts=2)
        if conf > 0:
            return sol, distinct
        best_sol, best_k = sol.copy(), distinct
        for _ in range(max_attempts):
            target = best_k - 1
            if target < calculate_lower_bound(self.G):
                break
            self.k_max = target
            tmp, (_, d2, c2) = self.run(max_restarts=2)
            if c2 == 0:
                best_sol, best_k = tmp.copy(), d2
            else:
                break
        return best_sol, best_k
