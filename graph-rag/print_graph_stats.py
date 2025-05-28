import networkx as nx
import sys
from collections import Counter

# Parse args
if len(sys.argv) != 2:
  print('Usage: python3 ./print_graph_stats.py <graphml-file-path>')
  sys.exit()

# Read in graph file
graph_file = sys.argv[1]
graph = nx.read_graphml(graph_file)

# Print simple statistics: graph type, number of nodes/edges, number of connected components
print(f'Graph is {'Directed' if graph.is_directed() else 'Undirected'}')
print(f'Number of nodes: {graph.number_of_nodes()}')
print(f'Number of edges: {graph.number_of_edges()}')

if graph.is_directed():
  print(f'Number of (weakly) connected components: {nx.number_weakly_connected_components(graph)}')
else:
  print(f'Number of connected components: {nx.number_connected_components(graph)}')

# Print average degree
degree_dict = dict(graph.degree())
avg_degree = sum(degree_dict.values()) / graph.number_of_nodes()
print(f'Average degree: {avg_degree}')

# Print nodes with high degrees
high_degree_nodes = Counter(degree_dict).most_common(10)
print()
print('10 highest-degree nodes:')
print(f'{"#":>2}  {"Node":<40} {"Degree":>6}')
print('-' * 50)
for num, (item, degree) in enumerate(high_degree_nodes, start=1):
  print(f'{num:>2}. {item:<40} {degree:>4}')

# Print frequent predicates (edge labels)
if graph.is_directed():
  # Undirected Microsoft GraphRAG graph has weights, but no labels on edges, so this analysis doesn't apply
  predicates = [data["relationship"] for _, _, data in graph.edges(data=True) if "relationship" in data]
  predicate_frequencies = Counter(predicates)
  frequent_predicates = predicate_frequencies.most_common(10)

  print()
  print('10 most common predicates (edge labels):')
  print(f'{"#":>2}  {"Predicate":<40} {"Frequency":<9}')
  print('-' * 50)
  for num, (item, count) in enumerate(frequent_predicates, start=1):
    print(f'{num:>2}. {item:<40} {count:>4}')
