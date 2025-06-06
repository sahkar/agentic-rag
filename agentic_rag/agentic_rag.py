from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from .base_rag import BaseRAG

class AgenticRAG(BaseRAG):
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


# load_dotenv()
# Settings.llm = OpenAI(model="gpt-4.1-mini")
# Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# class AgenticRAG(): 
#     def __init__(self):
#         self.storage_dir = "./storage"
#         self.data_dir = "./data"
#         self.query_engine_tools = []
#         self.agent = None
#         self.context = None

#     def load_rag(self): 
#         file_data = {}

#         docs = SimpleDirectoryReader(
#             self.data_dir, 
#             recursive=True, 
#             filename_as_id=True
#         ).load_data()

#         for doc in docs: 
#             file_name = doc.metadata.get("file_name", "unknown")
#             if file_name not in file_data: 
#                 file_data[file_name] = []
#             file_data[file_name].append(doc)
        
#         for file_name, file_docs in file_data.items(): 
#             print(f"Processing {file_name}")

#             try: 
#                 storage_context = StorageContext.from_defaults(
#                     persist_dir=f"{self.storage_dir}/{file_name}_vector"
#                 )
#                 index = load_index_from_storage(storage_context, index_id=f"{file_name}_vector")
#                 print(f"Index {file_name} loaded from storage")
#             except Exception as e: 
#                 print(f"Error loading index for {file_name}: {e}")
#                 print(f"Creating index for {file_name}")
                
#                 index = VectorStoreIndex.from_documents(
#                     documents=file_docs
#                 )
#                 index.storage_context.persist(persist_dir=f"{self.storage_dir}/{file_name}_vector")

#             if index: 
#                 engine = index.as_query_engine(similarity_top_k=5)
#                 tool = QueryEngineTool.from_defaults(
#                     query_engine=engine,
#                     name=file_name,
#                     description=f"Use this tool to answer questions about {file_name}"
#                 )
#                 self.query_engine_tools.append(tool)
    
#     def create_agent(self): 
#         self.agent = ReActAgent(
#             tools=self.query_engine_tools,
#             llm=Settings.llm,
#             system_prompt=self._get_system_prompt()
#         )
    
#     def create_context(self): 
#         self.context = Context(self.agent)
    
#     async def run_agent(self, question: str) -> str:
#         handler = self.agent.run(question, ctx=self.context)
        
#         async for ev in handler.stream_events():
#             if isinstance(ev, ToolCallResult):
#                 print(f"\033[36m\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}\033[0m")
#             if isinstance(ev, AgentStream):
#                 print(f"\033[33m{ev.delta}\033[0m", end="", flush=True)
                
#         response = await handler
#         return response
        
#     def add_message(self, message: str, role: str = "user") -> None: 
#         self.context.add_message(message, role)

#     def clear_context(self) -> None: 
#         self.context.clear()
        
#     def _get_system_prompt(self) -> str:
#         return """You are a helpful RAG agent that can answer questions about multiple documents. 
#         When answering questions:
#         1. First determine which document(s) are most relevant to the question
#         2. Use the appropriate tool(s) to search those documents
#         3. Always specify which document the information came from
#         4. If information comes from multiple documents, clearly indicate this
#         5. Keep all text in English
#         6. Provide exact quotes when relevant
#         7. Keep language as English""" 
