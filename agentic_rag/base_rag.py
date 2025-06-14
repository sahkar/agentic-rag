from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool


load_dotenv()
Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

class BaseRAG:
    def __init__(self):
        self.query_engine_tools = []
        self.agent:ReActAgent = None
        self.context = None

    def load_documents(self, input_dir: str, recursive: bool = True) -> None:
        """Load documents and create indices"""
        file_data = {}

        docs = SimpleDirectoryReader(
            input_dir, 
            recursive=recursive, 
            filename_as_id=True
        ).load_data()

        for doc in docs:
            file_name = doc.metadata.get("file_name", "unknown")
            if file_name not in file_data:
                file_data[file_name] = []
            file_data[file_name].append(doc)
    
        for file_name, file_docs in file_data.items():
            print(f"Processing {file_name}")
            
            try: 
                index = self._load_index(file_name)
            except Exception as e:
                print(f"Error loading index for {file_name}: {e}")
                index = self._process_document(file_name, file_docs)
            
            if index: 
                engine = index.as_query_engine(similarity_top_k=5)
                tool = QueryEngineTool.from_defaults(
                    query_engine=engine,
                    name=file_name,
                    description=f"Use this tool to answer questions about {file_name}"
                )
                self.query_engine_tools.append(tool)

    def _process_document(self, file_name: str, file_docs: list) -> None:
        """Process documents and create indices - to be implemented by subclasses"""
        raise NotImplementedError

    def _load_index(self, file_name: str) -> bool:
        """Load index from storage - to be implemented by subclasses"""
        raise NotImplementedError

    def create_agent(self) -> None:
        self.agent = ReActAgent.from_tools(
            tools=self.query_engine_tools,
            llm=Settings.llm, 
            system_prompt=self._get_system_prompt()
        )
        
    def _get_system_prompt(self) -> str:
        """Get system prompt - to be implemented by subclasses"""
        raise NotImplementedError

    def run_agent(self, question: str) -> str:
        return self.agent.stream_chat(question)  
    
    def add_message(self, message: str, role: str = "user") -> None:
        self.context.add_message(message, role)

    def clear_context(self) -> None:
        self.context.clear() 