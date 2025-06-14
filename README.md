# CSC 481 - Knowledge Based Systems Final Project
Names: Sahith Karra, Quinn Potter, Nipun Das
Instructor: Prof. Rodrigo Canaan

## Overview
This project involves exploraing and benchmarking two types of RAG systems : vector based RAG solutions and graph based RAG solutions. Our code base can be described as follows:

### /agentic_rag: 
This folder contains all of the code related to our LlamaIndex RAG pipelines including BaseRAG, VectorRAG, and GraphRAG. These classsesa are exposed via a relevant init file.

### /data: 
This folder can be used to upload relevant KBs which can be used for the RAG pipelines. 

### /graph-rag: 
This folder contains all of our native GraphRAG code including our native indexer. 

### /ui
This folder contains all of our Streamlit UI related work that is configured to demo the VectorRAG and GraphRAG pipelines. 

## Running the Code

```
pip install -r requirements.txt
```

### Llama Index
Runing the Streamlit UI for RAG demo
```
stremalut run ui/Vector_RAG.py
```



