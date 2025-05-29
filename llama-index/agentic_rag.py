from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    load_indices_from_storage
)
from llama_index.core.tools import QueryEngineTool
from base_rag import BaseRAG
import os

class AgenticRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.storage_dir = "./storage"

    def _process_documents(self, file_name: str, file_docs: list) -> None:
        """Process documents and create vector index"""
        # Create storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.storage_dir}/{file_name}"
        )
        
        # Check if index exists in storage
        index_path = os.path.join(self.storage_dir, f"{file_name}")
        if os.path.exists(index_path):
            try:
                print(f"Loading existing vector index for {file_name}")
                # Load all indices from storage
                indices = load_indices_from_storage(storage_context)
                if indices:
                    doc_index = indices[0]  # Get the first index
                    print(f"Successfully loaded vector index for {file_name}")
                else:
                    raise ValueError("No indices found in storage")
            except Exception as e:
                print(f"Error loading index: {e}")
                print(f"Creating new vector index for {file_name}")
                doc_index = self._create_new_index(file_docs, storage_context, file_name)
        else:
            print(f"No existing index found. Creating new vector index for {file_name}")
            doc_index = self._create_new_index(file_docs, storage_context, file_name)

        # Create vector query engine
        doc_engine = doc_index.as_query_engine(similarity_top_k=3, streaming=True)
        tool = QueryEngineTool.from_defaults(
            query_engine=doc_engine, 
            name=file_name, 
            description=f"Use this tool to answer questions about {file_name}", 
        )
        self.query_engine_tools.append(tool)

    def _create_new_index(self, file_docs: list, storage_context: StorageContext, file_name: str) -> VectorStoreIndex:
        """Helper method to create a new vector index"""
        doc_index = VectorStoreIndex.from_documents(
            documents=file_docs,
            storage_context=storage_context
        )
        # Set the index ID before persisting
        doc_index.set_index_id(f"{file_name}_vector")
        # Persist the index
        doc_index.storage_context.persist()
        return doc_index

    def _get_system_prompt(self) -> str:
        return """You are a helpful RAG agent that can answer questions about multiple documents. 
        When answering questions:
        1. First determine which document(s) are most relevant to the question
        2. Use the appropriate tool(s) to search those documents
        3. Always specify which document the information came from
        4. If information comes from multiple documents, clearly indicate this
        5. Keep all text in English
        6. Provide exact quotes when relevant
        7. Keep language as English""" 