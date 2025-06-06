from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import StorageContext, load_index_from_storage
from .base_rag import BaseRAG

class VectorRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.storage_dir = "./storage"

    def _load_index(self, file_name: str) -> bool:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.storage_dir}/{file_name}_vector"
        )

        index = load_index_from_storage(storage_context)
        print(f"Index {file_name} loaded from storage")
        return index
    
    def _process_document(self, file_name: str, file_docs: list) -> None:

        print(f"Creating index for {file_name}")
        index = VectorStoreIndex.from_documents(
            documents=file_docs
        )

        index.storage_context.persist(
            persist_dir=f"{self.storage_dir}/{file_name}_vector"
        )
        
        return index
    
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

