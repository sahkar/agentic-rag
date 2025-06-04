import os
import spacy
import json
import asyncio
import networkx as nx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="indexer/.env")
client = AsyncOpenAI()
nlp = spacy.load("en_core_web_sm")

# Load the GraphML file
G = nx.read_graphml("indexer/output/indexer_kg_pipeswitch.graphml")

# --- Helper Functions ---

def extract_candidate_nodes(question, graph):
    doc = nlp(question)
    candidates = set()
    graph_nodes = set(graph.nodes)
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if text in graph_nodes:
            candidates.add(text)
        else:
            for node in graph_nodes:
                if text.lower() in node.lower():
                    candidates.add(node)
    return list(candidates)

def subgraph_to_text(graph, center_node, depth=1):
    triples = []
    visited = set()
    queue = [(center_node, 0)]
    
    while queue:
        node, d = queue.pop(0)
        if d > depth or node in visited:
            continue
        visited.add(node)
        for neighbor in graph.successors(node):
            rel = graph[node][neighbor].get("relationship", "related to")
            triples.append(f"{node} {rel} {neighbor}")
            queue.append((neighbor, d + 1))
    
    return "\n".join(triples)

def get_context_from_nodes(graph, nodes, depth=1):
    return "\n".join(subgraph_to_text(graph, node, depth) for node in nodes)

async def answer_question_with_graph(graph, question, model="gpt-4o"):
    nodes = extract_candidate_nodes(question, graph)
    if not nodes:
        return "Could not find relevant context in the graph."

    context = get_context_from_nodes(graph, nodes, depth=2)

    # DEBUG
    print(context)

    prompt = f"""You are a helpful assistant answering questions about a research paper.

Use the following context extracted from a knowledge graph to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# --- Main ---

if __name__ == "__main__":
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = asyncio.run(answer_question_with_graph(G, question))
        print("\nAnswer:\n", answer)
