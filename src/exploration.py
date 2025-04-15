import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

class GraphExplorer:
    def __init__(self, filepath):
        
        self.filepath = filepath
        self.G = nx.Graph()
        self.n = 0
        self.m = 0

    def parse(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'p':
                    # problem line: p <#vertices> <#edges>
                    _, n, m = parts
                    self.n, self.m = int(n), int(m)
                    self.G.add_nodes_from(range(1, self.n + 1))
                elif parts[0] == 'e':
                    # edge line: e u v
                    _, u, v = parts
                    self.G.add_edge(int(u), int(v))

    def compute_basic_stats(self):
        # basic graph statistics.
        stats = {}
        stats['num_nodes'] = self.G.number_of_nodes()
        stats['num_edges'] = self.G.number_of_edges()
        stats['density']   = nx.density(self.G)
        degrees = [d for _, d in self.G.degree()]
        stats['avg_degree'] = sum(degrees) / len(degrees)
        stats['degree_histogram'] = Counter(degrees)
        stats['num_components']   = nx.number_connected_components(self.G)
        stats['component_sizes']  = sorted(
            (len(c) for c in nx.connected_components(self.G)),
            reverse=True
        )
        stats['avg_clustering'] = nx.average_clustering(self.G)
        return stats



    def draw_graph(self, nodes=None):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.G)
        if nodes:
            subgraph = self.G.subgraph(nodes)
            nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        else:
            nx.draw(self.G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        plt.show()

    def draw_random_subgraph(self, k=20, seed=None):
        
        # draws a random induced subgraph of k nodes to get a better visualization
        
        if seed is not None:
            random.seed(seed)
        nodes = random.sample(list(self.G.nodes()), k)
        self.draw_graph(nodes)


if __name__ == '__main__':
    # test1.txt
    explorer = GraphExplorer('data/benchmarks/test1.txt')
    explorer.parse()

    stats = explorer.compute_basic_stats()
    print("Basic stats for test1.txt:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Draw the graph
    explorer.draw_graph()

    # Draw a random subgraph of size 20
    explorer.draw_random_subgraph(k=20)
