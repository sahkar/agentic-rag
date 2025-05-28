import networkx as nx
import json
import os

with open("triples_relevant_riscv.json", "r") as file:
  triples = json.load(file)

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
print("Writing knowledge graph to indexer_kg_riscv.graphml...")
os.makedirs("output", exist_ok=True)
os.chdir("output")
nx.write_graphml(graph, "indexer_kg_riscv.graphml")
print("Complete")
