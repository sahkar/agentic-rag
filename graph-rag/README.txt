1. Use indexer to get triples
    - run group_chunks.py with selected paper (have to set paper manually in file)
    - run extract_triples.py on clustered chunks output from group_chunks.py (have to set paper manually in file)
2. Create graph from triples
    - run generate_graph.py with selected paper set manually in file
3. Do RAG on graph
    - run graphRAG.py (set graphml file manually, currently they're in indexer/outputs)
    - enter query
        - prints gathered subgraph triples. These are sent as context to the LLM. This context may be extensive if you ask about a central 
        entity, ex. querying "what is pipeswitch" on the pipeswitch paper will give hundreds of nodes for context.
        - outputs LLM's actual response to the context and query

might need spacy, sklearn, BERT, and other imports from these scripts (I just used the version that pip install chose)
definitely need valid openai api key