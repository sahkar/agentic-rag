from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
import asyncio
load_dotenv()
Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

class AgenticRAG: 
    def __init__(self): 
        self.query_engine_tools = []
    
    def load_index(self, input_dir: str, recursive: bool = True) -> None:
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
            doc_index = VectorStoreIndex.from_documents(file_docs)
            persist_dir = f"./storage/{file_name}"
            doc_index.storage_context.persist(persist_dir=persist_dir)

            doc_engine = doc_index.as_query_engine(similarity_top_k=3)
            tool = QueryEngineTool.from_defaults(
                query_engine=doc_engine, 
                name=file_name, 
                description=f"Use this tool to answer questions about {file_name}"
            )
            self.query_engine_tools.append(tool)

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
            system_prompt="""You are a helpful RAG agent that can answer questions about multiple documents. 
            When answering questions:
            1. First determine which document(s) are most relevant to the question
            2. Use the appropriate tool(s) to search those documents
            3. Always specify which document the information came from
            4. If information comes from multiple documents, clearly indicate this
            5. Keep all text in English
            6. Provide exact quotes when relevant
            7. Keep language as English"""
        )
        
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
    
    def add_message(self, message: str, role: str = "user") -> None:
        self.context.add_message(message, role)

    def clear_context(self) -> None:
        self.context.clear()

if __name__ == "__main__":
    rag = AgenticRAG()
    rag.load_rag()
    rag.create_agent()
    rag.create_context()
    question = "What is pipeswitch. What is vLLM?"
    response = asyncio.run(rag.run_agent(question))
    print(f"\n\nQuestion: {question}\nResponse: {response}")
