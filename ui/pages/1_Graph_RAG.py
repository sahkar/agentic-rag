import streamlit as st
import sys
import os
import tempfile
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agentic_rag import GraphRAG, BaseRAG


if __name__ == "__main__":
    st.title("Graph RAG")
    files = st.file_uploader("Upload files", type=["pdf", "txt", "docx", "doc"], accept_multiple_files=True)

    if files:
        if "temp_dir" not in st.session_state:
            st.session_state["temp_dir"] = tempfile.mkdtemp()
        
        temp_dir = st.session_state["temp_dir"]
        
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        

    if "graph" not in st.session_state:
        st.session_state["graph"] = GraphRAG()
        rag:BaseRAG = st.session_state["graph"]
        
        if files and "temp_dir" in st.session_state:
            rag.load_documents(st.session_state["temp_dir"])
        
        rag.create_agent()

    rag = st.session_state["graph"]
    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = []

    for message in st.session_state.graph_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Enter a message..."):
        st.session_state.graph_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.run_agent(prompt)

                st.write_stream(response.response_gen)
                st.session_state.graph_messages.append({"role": "graph", "content": response.response})