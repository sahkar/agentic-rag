from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult, AgentStream

load_dotenv()
Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

class BaseRAG:
    def __init__(self):
        self.query_engine_tools = []
        self.agent = None
        self.context = None

    def load_index(self, input_dir: str, recursive: bool = True) -> None:
        """Load documents and create indices"""
        docs_by_file = {}
        docs = SimpleDirectoryReader(
            input_dir, 
            recursive=recursive, 
            filename_as_id=True
        ).load_data()

        for doc in docs:
            file_name = doc.metadata.get("file_name", "unknown")
            if file_name not in docs_by_file:
                docs_by_file[file_name] = []
            docs_by_file[file_name].append(doc)
    
        for file_name, file_docs in docs_by_file.items():
            print(f"Processing {file_name}")
            self._process_documents(file_name, file_docs)

    def _process_documents(self, file_name: str, file_docs: list) -> None:
        """Process documents and create indices - to be implemented by subclasses"""
        raise NotImplementedError

    def _load_existing_indices(self) -> bool:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir='./storage/'
            )
            indices = load_index_from_storage(storage_context)
            return True
        except:
            return False
    
    def load_rag(self) -> None:
        if not self._load_existing_indices():
            self.load_index("./data")
        else:
            print("RAG already loaded")

    def create_agent(self) -> None:
        self.agent = ReActAgent(
            tools=self.query_engine_tools,
            llm=Settings.llm,
            system_prompt=self._get_system_prompt(), 
        )
        
    def _get_system_prompt(self) -> str:
        """Get system prompt - to be implemented by subclasses"""
        raise NotImplementedError

    def create_context(self) -> Context:
        self.context = Context(self.agent)

    async def run_agent(self, question: str) -> str:
        handler = self.agent.run(question, ctx=self.context)
        
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)
                
        response = await handler
        return response

    # def run_agent(self, question: str) -> str:
    #     return self.agent.run(question, ctx=self.context)
    
    def add_message(self, message: str, role: str = "user") -> None:
        self.context.add_message(message, role)

    def clear_context(self) -> None:
        self.context.clear() 