
import networkx as nx
import math




def calculate_upper_bound(G:nx.Graph) -> int:
        """Calculate an upper bound using a greedy coloring."""
        coloring = {}
        order = list(G.nodes())
        for v in order:
            # Find the lowest color not used by neighbors
            used_colors = {coloring.get(u) for u in G.neighbors(v) if u in coloring}
            color = 1
            while color in used_colors:
                color += 1
            coloring[v] = color
        
        
        return max(coloring.values()) if coloring else 0
 
def calculate_lower_bound(G:nx.Graph) -> int:
        """Calculate a lower bound for the chromatic number using clique size."""
        # Find a maximal clique as a lower bound (not necessarily maximum)
        # Using a greedy algorithm for speed
        max_clique_size = 1
        for start_node in G.nodes():
            clique = {start_node}
            potential_nodes = set(G.nodes())
            
            while potential_nodes:
                # Find nodes connected to all nodes in current clique
                candidates = potential_nodes.copy()
                for node in clique:
                    candidates &= set(G.neighbors(node))
                
                if not candidates:
                    break
                    
                # Add highest degree node
                next_node = max(candidates, key=lambda x: G.degree(x))
                clique.add(next_node)
                potential_nodes.remove(next_node)
            
            max_clique_size = max(max_clique_size, len(clique))
        
        # The largest degree + 1 is also a lower bound
        max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
        
        return max(max_clique_size, (max_degree + 1) // 2)

