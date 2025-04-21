import random
import networkx as nx
from typing import List, Tuple
from collections import Counter

class BSOColoring:
    def __init__(
        self,
        G: nx.Graph,                 
        k_max: int,                  # maximum number of colors to use
        n_bees: int = 30,            # size of the swarm
        n_chance: int = 3,           # duplicate threshold before injecting diversity
        max_iter: int = 500,         # maximum iterations
        max_steps: int = 15,         # maximum steps for local search
        flip: int = 5,               # flip parameter for search area determination
        alpha: int = None,           # conflict penalty (defaults to |E|+1)
        seed: int = None,            # random seed
    ):
        
        if seed is not None:
            random.seed(seed)

        self.G = G
        self.n = G.number_of_nodes()
        self.k_max = k_max
        self.n_bees = n_bees
        self.n_chance = n_chance
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.flip = flip
        self.alpha = alpha or (G.number_of_edges() + 1)
        
        # Initialize taboo list to store previously visited reference solutions
        self.taboo_list = []
        
        # Initialize reference solution
        self.S_ref = self.random_coloring()

    def random_coloring(self) -> List[int]:
        # Generate a random coloring (1..k_max) for n nodes
        return [random.randint(1, self.k_max) for _ in range(self.n)]

    def fitness(self, S: List[int]) -> int:
        # Calculate the fitness of a coloring
        # fitness = alpha * conflicts + distinct_colors
        
        conflicts = sum(1 for u, v in self.G.edges() if S[u-1] == S[v-1])
        distinct = len(set(S))
        return self.alpha * conflicts + distinct
    
    def determine_search_area(self) -> List[List[int]]:
        # Implementation of the search area determination
        # Based on the pseudocode provided
        search_area = []
        h = 0
        
        while len(search_area) < self.n_bees and h < self.flip:
            s = self.S_ref.copy()
            p = 0
            while self.flip * p + h < self.n:
                # Change (flip) the color at position flip*p+h
                current_color = s[self.flip * p + h]
                # Choose a new color different from the current one
                new_color = random.choice([c for c in range(1, self.k_max+1) if c != current_color])
                s[self.flip * p + h] = new_color
                p += 1
            
            search_area.append(s)
            h += 1
            
        return search_area

    def hamming(self, A: List[int], B: List[int]) -> int:
        # Calculate the Hamming distance between two colorings
        # Hamming distance = number of positions with different colors
        return sum(1 for x, y in zip(A, B) if x != y)
    
    def diversity(self, solution: List[int]) -> int:
        # Calculate the diversity of a solution relative to the taboo list
        # Diversity is measured as the minimum distance to any solution in the taboo list
        if not self.taboo_list:
            return 0
        
        return min(self.hamming(solution, taboo_sol) for taboo_sol in self.taboo_list)

    def inject_diversity(self, table: List[List[int]]) -> None:
        # Inject diversity into the population by replacing duplicates
        
        # Count occurrences
        cnt = Counter(tuple(s) for s in table)
        
        # Find duplicates
        for sol, occ in cnt.items():
            if occ > self.n_chance:
                # Find all positions where this solution occurs
                positions = [i for i, S in enumerate(table) if tuple(S) == sol]
                
                # Leave first n_chance, replace the rest
                for pos in positions[self.n_chance:]:
                    # Pick the candidate in table farthest from all others
                    divers = max(
                        table,
                        key=lambda T: sum(self.hamming(T, U) for U in table)
                    )
                    table[pos] = divers.copy()
    
    def local_search(self, solution: List[int]) -> List[int]:
        # Perform local search from the given solution
        best_solution = solution.copy()
        best_fitness = self.fitness(solution)
        
        for _ in range(self.max_steps):
            # Choose a random node and change its color
            node = random.randrange(self.n)
            current_color = solution[node]
            new_color = random.choice([c for c in range(1, self.k_max+1) if c != current_color])
            
            # Create new solution by changing one color
            new_solution = solution.copy()
            new_solution[node] = new_color
            
            # Calculate new fitness
            new_fitness = self.fitness(new_solution)
            
            # If better, update best solution
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                solution = new_solution.copy()
        
        return best_solution
    
    def select_new_reference(self, dance_table: List[List[int]], chances: int) -> Tuple[List[int], int]:
        # Select new reference solution based on quality and diversity
        best_quality = min(dance_table, key=lambda s: self.fitness(s))
        best_quality_fitness = self.fitness(best_quality)
        
        # Check if we have improvement compared to previous S_ref
        if best_quality_fitness < self.fitness(self.S_ref):
            # We have improvement, return best quality
            return best_quality, self.n_chance
        else:
            # No improvement, decrease chances
            chances -= 1
            if chances > 0:
                # Still have chances, return best quality
                return best_quality, chances
            else:
                # No more chances, return best diversity
                best_diversity = max(dance_table, key=lambda s: self.diversity(s))
                return best_diversity, self.n_chance

    def run(self) -> Tuple[List[int], int]:
        # Main BSO loop. Returns the best coloring found and its fitness.
        chances = self.n_chance
        
        for _ in range(self.max_iter):
            # Add current reference solution to taboo list
            self.taboo_list.append(self.S_ref.copy())
            
            # Determine search area from reference solution
            search_area = self.determine_search_area()
            
            # Ensure we have exactly n_bees solutions in the search area
            # If not enough, fill with random variants of S_ref
            while len(search_area) < self.n_bees:
                new_sol = self.S_ref.copy()
                for _ in range(random.randint(1, self.n // 10)):
                    idx = random.randrange(self.n)
                    new_sol[idx] = random.randint(1, self.k_max)
                search_area.append(new_sol)
            
            # Assign solutions to bees and perform local search
            dance_table = []
            for bee_solution in search_area:
                best_local = self.local_search(bee_solution)
                dance_table.append(best_local)
            
            # Inject diversity into dance table
            self.inject_diversity(dance_table)
            
            # Select new reference solution
            self.S_ref, chances = self.select_new_reference(dance_table, chances)
            
            # Early stop if valid (no conflicts)
            if self.fitness(self.S_ref) < self.alpha:  # zero conflicts => fitness < alpha
                break
        
        return self.S_ref, self.fitness(self.S_ref)