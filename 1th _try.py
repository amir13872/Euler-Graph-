
file_path = "/Desktop/data/video_2.pdf"

import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations
import sympy as sp

# --- Helper functions / classes -------------------------------------------------

class RuleGraph:
    def __init__(self):
        self.G = nx.Graph()
    
    def add_nodes(self, nodes):
        """
        nodes: iterable of (node_id, {attr: value, ...})
        """
        for nid, attrs in nodes:
            self.G.add_node(nid, **(attrs or {}))

    def add_edges_distance(self, coord_attr=('x','y'), threshold=1.0, inclusive=True):
        """Add undirected edge between u and v if Euclidean distance between coordinates < threshold"""
        nodes = list(self.G.nodes(data=True))
        for i in range(len(nodes)):
            u, au = nodes[i][0], nodes[i][1]
            if coord_attr[0] not in au or coord_attr[1] not in au:
                continue
            for j in range(i+1, len(nodes)):
                v, av = nodes[j][0], nodes[j][1]
                if coord_attr[0] not in av or coord_attr[1] not in av:
                    continue
                dx = au[coord_attr[0]] - av[coord_attr[0]]
                dy = au[coord_attr[1]] - av[coord_attr[1]]
                d = math.hypot(dx, dy)
                if (d <= threshold if inclusive else d < threshold):
                    self.G.add_edge(u, v, weight=d)
    
    def add_edges_symbolic(self, expr_str, attr_symbols):
        """
        expr_str: e.g. "a + b - 3" or "a*b - 1" where a corresponds to attribute name attr_symbols[0], etc.
                  Add edge between u and v if expr evaluated with u[attr0] and v[attr1] equals 0.
        attr_symbols: tuple/list of attribute names in order (attr_for_u, attr_for_v)
        """
        # create sympy symbols to parse
        sym_a, sym_b = sp.symbols('a b')
        expr = sp.sympify(expr_str)
        for u, v in combinations(self.G.nodes(), 2):
            au = self.G.nodes[u]
            av = self.G.nodes[v]
            if attr_symbols[0] in au and attr_symbols[1] in av:
                val = expr.subs({sym_a: au[attr_symbols[0]], sym_b: av[attr_symbols[1]]})
                try:
                    is_zero = sp.N(val) == 0
                except Exception:
                    is_zero = (val == 0)
                if is_zero:
                    self.G.add_edge(u, v, rule=f"sym:{expr_str}")
    
    def add_edges_by_predicate(self, predicate):
        """
        predicate: function(u, v, attrs_u, attrs_v) -> True/False
        """
        for u, v in combinations(self.G.nodes(), 2):
            au = self.G.nodes[u]
            av = self.G.nodes[v]
            try:
                if predicate(u, v, au, av):
                    self.G.add_edge(u, v)
            except Exception:
                pass
    
    # --- analyses ---------------------------------------------------------------
    def degrees(self):
        return dict(self.G.degree())
    
    def neighbors(self, node):
        return list(self.G.neighbors(node))
    
    def is_connected(self):
        return nx.is_connected(self.G) if self.G.number_of_nodes()>0 else True
    
    def is_eulerian(self):
        return nx.is_eulerian(self.G) if self.G.number_of_nodes()>0 else False
    
    def is_semi_eulerian(self):
        # exactly two vertices of odd degree and connected (or Eulerian)
        odd = [v for v,d in self.G.degree() if d%2==1]
        return (len(odd) == 2 and self.is_connected()) or self.is_eulerian()
    
    def complement(self):
        return nx.complement(self.G)
    
    def havel_hakimi(self, deg_sequence):
        # networkx provides is_graphical (Havel-Hakimi), use it
        return nx.is_graphical(list(deg_sequence), method='hh')
    
    def greedy_dominating_set(self):
        # simple greedy algorithm (approximation)
        H = self.G.copy()
        dom = set()
        uncovered = set(H.nodes())
        while uncovered:
            # pick node that covers most uncovered nodes (including itself)
            best = None
            best_cov = -1
            for v in H.nodes():
                cov = set([v]) | set(H.neighbors(v))
                cov_uncovered = len(cov & uncovered)
                if cov_uncovered > best_cov:
                    best_cov = cov_uncovered
                    best = v
            if best is None:
                break
            dom.add(best)
            covered = set([best]) | set(H.neighbors(best))
            uncovered -= covered
        return dom

    def find_paths(self, source, target, max_length=None):
        # list simple paths up to max_length (edge count)
        if source not in self.G or target not in self.G:
            return []
        paths = list(nx.all_simple_paths(self.G, source, target, cutoff=max_length))
        return paths
    
    def cycles(self):
        # simple cycle basis
        return nx.cycle_basis(self.G)
    
    # plotting
    def plot(self, figsize=(8,6), with_labels=True, savepath='graph_demo.png'):
        pos = {}
        # if nodes have x,y use them; otherwise use spring layout
        use_coords = all(('x' in self.G.nodes[n] and 'y' in self.G.nodes[n]) for n in self.G.nodes())
        if use_coords:
            pos = {n: (self.G.nodes[n]['x'], self.G.nodes[n]['y']) for n in self.G.nodes()}
        else:
            pos = nx.spring_layout(self.G, seed=1)
        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, with_labels=with_labels, node_size=300)
        plt.title("Graph visualization (rule-built)")
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        plt.show()
        return savepath

# --- Demo: build graphs using rules and analyze --------------------------------

rg = RuleGraph()

# Example: add nodes with coordinates and an integer attribute 'value'
nodes = [
    ('A', {'x':0,'y':0, 'val': 1}),
    ('B', {'x':1,'y':0, 'val': 2}),
    ('C', {'x':2,'y':0, 'val': -1}),
    ('D', {'x':0.5,'y':1.2, 'val': 0}),
    ('E', {'x':2.4,'y':0.8, 'val': 1}),
    ('F', {'x':-1,'y':0.2, 'val': 2}),
]
rg.add_nodes(nodes)

# Rule 1: connect by geometric rule (distance <= 1.5)
rg.add_edges_distance(coord_attr=('x','y'), threshold=1.5, inclusive=True)

# Rule 2: symbolic rule between 'val' attributes: add edge if val_u + val_v - 1 == 0  (i.e., val_u + val_v == 1)
rg.add_edges_symbolic("a + b - 1", ('val','val'))

# Rule 3: custom predicate example: connect if x coordinate difference < 0.7
rg.add_edges_by_predicate(lambda u,v,au,av: abs(au.get('x',0)-av.get('x',0)) < 0.7)

# Now analyze
degrees = rg.degrees()
neighbors_of_A = rg.neighbors('A')
is_conn = rg.is_connected()
is_euler = rg.is_eulerian()
is_semi = rg.is_semi_eulerian()
dom_set = rg.greedy_dominating_set()
complement_graph = rg.complement()
paths_A_C = rg.find_paths('A','C', max_length=4)
cycles = rg.cycles()

# Havel-Hakimi demo
seq1 = [3,3,2,2,1,1]  # example sequence
is_graphical_seq1 = rg.havel_hakimi(seq1)

# Solve a sample symbolic equation (demonstration of "solve equations"): solve x + y = 3 with integer restriction
x,y = sp.symbols('x y', integer=True)
solutions_cut = sp.solve([x + y - 3], [x,y], dict=True)
# show small set of integer pairs for demonstration
int_solutions = [(i, 3-i) for i in range(-2,6)]

# Plot and save
img_path = rg.plot(savepath='graph_demo.png')

# Summary output in a dict
output = {
    'file_used': file_path,
    'nodes': list(rg.G.nodes(data=True)),
    'edges': list(rg.G.edges(data=True)),
    'degrees': degrees,
    'neighbors_of_A': neighbors_of_A,
    'is_connected': is_conn,
    'is_eulerian': is_euler,
    'is_semi_eulerian': is_semi,
    'dominating_set_greedy': list(dom_set),
    'complement_edges_sample': list(complement_graph.edges())[:10],
    'paths_A_to_C_up_to_4': paths_A_C,
    'cycles_basis': cycles,
    'havel_hakimi_seq1_graphical': is_graphical_seq1,
    'symbolic_demo_solutions_sample': int_solutions,
    'saved_image': img_path
}

output

