import networkx as nx
import json
import os

with open("triples_relevant.json", "r") as file:
  triples = json.load(file)

graph = nx.DiGraph()

for triple in triples:
  subject = triple["subject"]
  predicate = triple["predicate"]
  object = triple["object"]

  if None in [subject, predicate, object]: continue

  if subject not in graph:
    graph.add_node(subject)
  
  if object not in graph:
    graph.add_node(object)

  graph.add_edge(subject, object, relationship="predicate")

print("Writing knowledge graph to indexer_kg.graphml...")
os.makedirs("output", exist_ok=True)
os.chdir("output")
nx.write_graphml(graph, "indexer_kg.graphml")