from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


class AgenticRAG:
    def __init__(self):
        load_dotenv()
        
        documents = SimpleDirectoryReader('./data').load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)

        summary_query_engine = summary_index.as_query_engine(
            response_model="tree_summary", 
            streaming=True
        )

        vector_query_engine = vector_index.as_query_engine(
            streaming=True
        )

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine, 
            description="Useful for summarizing the papers provided", 
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine, 
            description="Useful for retrieing specific context related to the papers provided"
        )

        self.query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(), 
            query_engine_tools=[summary_tool, vector_tool], 
            verbose=True
        )

    def query(self, prompt):
        response = self.query_engine.query(prompt)
        return response
