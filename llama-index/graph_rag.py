from llama_index.core import (
    StorageContext,
    KnowledgeGraphIndex,
    load_index_from_storage,
    load_indices_from_storage
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.tools import QueryEngineTool
import networkx as nx
from base_rag import BaseRAG
import os

class GraphRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.graph_store = SimpleGraphStore()
        self.storage_dir = "./storage"

    def _process_documents(self, file_name: str, file_docs: list) -> None:
        """Process documents and create both vector and graph indices"""
        # Create storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.storage_dir}/{file_name}_graph"
        )
        
        # Check if index exists in storage
        index_path = os.path.join(self.storage_dir, f"{file_name}_graph")
        if os.path.exists(index_path):
            try:
                print(f"Loading existing graph index for {file_name}")
                # Load all indices from storage
                indices = load_indices_from_storage(storage_context)
                if indices:
                    graph_index = indices[0]  # Get the first index
                    print(f"Successfully loaded graph index for {file_name}")
                else:
                    raise ValueError("No indices found in storage")
            except Exception as e:
                print(f"Error loading index: {e}")
                print(f"Creating new graph index for {file_name}")
                graph_index = self._create_new_index(file_docs, storage_context, file_name)
        else:
            print(f"No existing index found. Creating new graph index for {file_name}")
            graph_index = self._create_new_index(file_docs, storage_context, file_name)

        graph_engine = graph_index.as_query_engine(
            include_text=True,
            retriever_mode="keyword",
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5
        )
        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_engine,
            name=f"{file_name}_graph",
            description=f"Use this tool to answer questions about {file_name} using knowledge graph relationships"
        )
        self.query_engine_tools.append(graph_tool)

    def _create_new_index(self, file_docs: list, storage_context: StorageContext, file_name: str) -> KnowledgeGraphIndex:
        """Helper method to create a new knowledge graph index"""
        graph_index = KnowledgeGraphIndex.from_documents(
            documents=file_docs,
            max_triplets_per_chunk=3,
            storage_context=storage_context,
            include_embeddings=True
        )
        # Set the index ID before persisting
        graph_index.set_index_id(f"{file_name}_graph")
        # Persist the index
        graph_index.storage_context.persist()
        return graph_index

    def _get_system_prompt(self) -> str:
        return """You are a helpful RAG agent that can answer questions using both vector search and knowledge graph relationships. 
        When answering questions:
        1. First determine which document(s) are most relevant to the question
        2. Use both vector search and knowledge graph tools to gather information
        3. Consider relationships between entities from the knowledge graph
        4. Always specify which document and method (vector/graph) the information came from
        5. If information comes from multiple sources, clearly indicate this
        6. Keep all text in English
        7. Provide exact quotes when relevant"""
