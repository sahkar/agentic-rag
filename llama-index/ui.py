import streamlit as st
from agentic_rag import AgenticRAG
from graph_rag import GraphRAG
import asyncio
rag_types = {
    "Agentic RAG": AgenticRAG,
    "Graph RAG": GraphRAG
}
from llama_index.core.agent.workflow import ToolCallResult, AgentStream

st.title("RAG Demo")

rag_type = st.selectbox("Select RAG Type", list(rag_types.keys()))

rag = rag_types[rag_type]()
rag.load_rag()
rag.create_agent()
rag.create_context()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Enter a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    handler = rag.run_agent(prompt)
    st.write_stream(handler.response_gen)
    response = handler.response

    print(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)