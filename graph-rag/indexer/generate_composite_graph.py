import networkx as nx
import json
import os

# currently options are pipeswitch, riscv, vLLM
input_papers = ['vLLM', 'riscv']
output_file_name = "indexer_kg_composite.graphml"

triples = []

for input_paper in input_papers:
  with open(f"triples_relevant_{input_paper}.json", "r") as file:
    paper_triples = json.load(file)
    triples.extend(paper_triples)

graph = nx.DiGraph()

num_triples = 0

for triple in triples:
  subject = triple["subject"]
  predicate = triple["predicate"]
  object = triple["object"]

  is_valid = True
  for id in [subject, predicate, object]:
    if id is None or len(id) == 0:
      is_valid = False
      continue

  # if we have any invalid IDs, don't process this triple
  if not is_valid:
    continue

  num_triples += 1

  if subject not in graph:
    graph.add_node(subject)
  
  if object not in graph:
    graph.add_node(object)

  graph.add_edge(subject, object, relationship=predicate)

print(f"Num triples: {num_triples}")
print(f"Writing knowledge graph to {output_file_name}...")
os.makedirs("output", exist_ok=True)
os.chdir("output")
nx.write_graphml(graph, output_file_name)
print("Complete")
