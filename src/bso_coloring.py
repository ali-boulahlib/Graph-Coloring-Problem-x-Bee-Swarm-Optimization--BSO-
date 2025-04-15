import random
import networkx as nx
from typing import List, Tuple
from collections import Counter

class BSOColoring:
    def __init__(
        self,
        G: nx.Graph,                 
        k_max: int,                     # maximum number of colors to use
        n_bees: int = 30,               # size of the swarm
        n_neighbors: int = 10,          # neighborhood size per bee
        n_chance: int = 3,              # duplicate threshold before injecting diversity
        max_iter: int = 500,            # maximum iterations
        alpha: int = None,              # conflict penalty (defaults to |E|+1)
        seed: int = None,               # random seed
    ):
        
        if seed is not None:
            random.seed(seed)

        self.G = G
        self.n = G.number_of_nodes()
        self.k_max = k_max
        self.n_bees = n_bees
        self.n_neighbors = n_neighbors
        self.n_chance = n_chance
        self.max_iter = max_iter
        self.alpha = alpha or (G.number_of_edges() + 1)

        # Initialize reference solution
        self.S_ref = self.random_coloring()

    def random_coloring(self) -> List[int]:
        # Generate a random coloring (1..k_max) for n nodes
        return [random.randint(1, self.k_max) for _ in range(self.n)]

    def fitness(self, S: List[int]) -> int:
        #calculate the fitness of a coloring
        # fitness = alpha * conflicts + distinct_colors
        
        conflicts = sum(1 for u,v in self.G.edges() if S[u-1] == S[v-1])
        distinct = len(set(S))
        return self.alpha * conflicts + distinct

    def flip_one_node(self, S: List[int]) -> List[int]:
    
        # pick a random node and recolor it 
        i = random.randrange(self.n)
        new_color = random.choice([c for c in range(1, self.k_max+1) if c != S[i]])
        S2 = S.copy()
        S2[i] = new_color
        return S2

    def hamming(self, A: List[int], B: List[int]) -> int:
        # calculate the Hamming distance between two colorings
        # Hamming distance = number of positions with different colors
        return sum(1 for x,y in zip(A,B) if x != y)

    def inject_diversity(self, table: List[List[int]]) -> None:
        # Inject diversity into the population by replacing duplicates

        # count occurrences
        cnt = Counter(tuple(s) for s in table)
        # find duplicates
        for sol, occ in cnt.items():
            if occ > self.n_chance:
                # find all positions where this sol occurs
                positions = [i for i,S in enumerate(table) if tuple(S) == sol]
                # leave first n_chance, replace the rest
                for pos in positions[self.n_chance:]:
                    # pick the candidate in table farthest from all others
                    divers = max(
                        table,
                        key=lambda T: sum(self.hamming(T, U) for U in table)
                    )
                    table[pos] = divers.copy()

    def run(self) -> Tuple[List[int], int]:
       
        # Main BSO loop. Returns the best coloring found and its fitness.
       
        for iteration in range(self.max_iter):
            dance = []

            # each bee proposes an improved candidate
            for _ in range(self.n_bees):
                s0 = self.flip_one_node(self.S_ref)
                best_local = s0
                f_best = self.fitness(s0)

                for _ in range(self.n_neighbors):
                    s1 = self.flip_one_node(s0)
                    f1 = self.fitness(s1)
                    if f1 < f_best:
                        best_local, f_best = s1, f1

                dance.append(best_local)

            # diversity injection
            self.inject_diversity(dance)

            # pick best from dance
            dance.sort(key=self.fitness)
            self.S_ref = dance[0]

            # early stop if valid
            if self.fitness(self.S_ref) < self.alpha:  # zero conflicts => fitness < alpha
                break

        return self.S_ref, self.fitness(self.S_ref)
