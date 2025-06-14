# CSC 481 - Knowledge Based Systems Final Project
Names: Sahith Karra, Quinn Potter, Nipun Das <br>
Instructor: Dr. Rodrigo Canaan

## Overview
This project involves exploring and benchmarking two types of RAG systems : vector based RAG solutions and graph based RAG solutions. Our code base can be described as follows:

### /agentic_rag: 
This folder contains all of the code related to our LlamaIndex RAG pipelines including BaseRAG, VectorRAG, and GraphRAG. These classsesa are exposed via a relevant init file.

### /data: 
This folder can be used to upload relevant KBs which can be used for the RAG pipelines. 

### /graph-rag: 
This folder contains all of our native GraphRAG code including our native indexer. 

### /ui
This folder contains all of our Streamlit UI related work that is configured to demo the VectorRAG and GraphRAG pipelines. 

## Running the Code

Run the following line to install all necessary dependencies:
```
pip install -r requirements.txt
```

If any dependencies are not included in the requirements.txt folder, they can be installed with a simple `pip install <dependency-name>`.

### Llama Index
Runing the Streamlit UI for RAG demo
```
streamlit run ui/Vector_RAG.py
```

### Native GraphRAG Pipeline
There are a few steps to run the native GraphRAG pipeline, listed below.
Note that for the files for this can be found in the ./graph-rag/indexer directory.
To run the indexer and to use the GraphRAG system when querying, an OpenAPI key needs to be
provided through a .env file. There is an example .env.sample file in this repository, but note
that the .env file used by the GraphRAG implementation needs to go inside the
./graph-rag/indexer directory.

1. Creating clustered chunks (specify which paper in script)
```
python3 ./group_chunks.py
```

2. Extracting triples (specify which paper in script)
```
python3 ./extract_chunks.py
```

3. Build graph (specify which paper in script)
If you would like to build a graph from multiple papers, you can run the
generate_composite_graph.py script instead (the list of papers used is in the file).
```
python3 ./generate_graph.py
```

4. Prompt the LLM using GraphRAG
Specify which graphml file to read from in the script.
```
python3 ./graphRAG.py
```

## Generating results
For this project, most of our results were obtained by running either the LlamaIndex RAG
implementation or our own native GraphRAG pipeline and comparing outputs. The instructions for
running either of those RAG systems is above. If you want to determine graph statistics for a file

```
python3 ./print_graph_stats.py <graphml-file-path>
```

The results from comparing our native indexer graph to the LlamaIndex-generated graph are included
below:

| Attribute                       | LlamaIndex | Our Indexer |
|---------------------------------|------------|-------------|
| # Nodes                         | 106        | 975         |
| # Edges                         | 94         | 729         |
| # Weakly Connected Components   | 15         | 261         |
| Average Node Degree             | 1.712      | 1.495       |
