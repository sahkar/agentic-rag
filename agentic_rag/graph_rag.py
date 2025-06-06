from llama_index.core import (
    StorageContext,
    KnowledgeGraphIndex,
    load_index_from_storage
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.tools import QueryEngineTool
from .base_rag import BaseRAG


class GraphRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.storage_dir = "./storage"
        self.graph_store = SimpleGraphStore()

    def _load_index(self, file_name: str) -> bool:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.storage_dir}/{file_name}_graph", 
            graph_store=self.graph_store
        )
        index = load_index_from_storage(storage_context, max_triplets_per_chunk=10, verbose=True)
        print(f"Index {file_name} loaded from storage")
        return index

    def _process_document(self, file_name: str, file_docs: list) -> None:
        print(f"Creating index for {file_name}")

        storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store
        )

        index = KnowledgeGraphIndex.from_documents(
            documents=file_docs,
            max_triplets_per_chunk=10,
            include_embeddings=True,
            include_text=True, 
            show_progress=True,
            storage_context=storage_context
        )

        index.storage_context.persist(
            persist_dir=f"{self.storage_dir}/{file_name}_graph"
        )
        
        return index

    def _get_system_prompt(self) -> str:
        return """You are a helpful RAG agent that can answer questions using both vector search and knowledge graph relationships. 
        When answering questions:
        1. First determine which document(s) are most relevant to the question
        2. Use both vector search and knowledge graph tools to gather information
        3. Consider relationships between entities from the knowledge graph
        4. Always specify which document and method (vector/graph) the information came from
        5. If information comes from multiple sources, clearly indicate this
        6. Keep all text in English
        7. Provide exact quotes when relevant
        8. When discussing results or findings, make sure to include both the relationships and the specific details"""