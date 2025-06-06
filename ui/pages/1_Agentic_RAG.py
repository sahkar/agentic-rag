import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from agentic_rag import AgenticRAG, BaseRAG

if __name__ == "__main__":

    st.title("Agentic RAG")

    files = st.file_uploader("Upload files", type=["pdf", "txt", "docx", "doc"], accept_multiple_files=True)

    if files:
        # Create a temporary directory for uploaded files
        if "temp_dir" not in st.session_state:
            st.session_state["temp_dir"] = tempfile.mkdtemp()
        
        temp_dir = st.session_state["temp_dir"]
        
        # Save uploaded files to temporary directory
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        
        st.toast(f"Uploaded {len(files)} files")

    if "agentic" not in st.session_state:
        st.session_state["agentic"] = AgenticRAG()
        rag:BaseRAG = st.session_state["agentic"]
        
        # Use uploaded files directory if available, otherwise use default data directory
        if files and "temp_dir" in st.session_state:
            rag.load_documents(st.session_state["temp_dir"])
        else:
            rag.load_documents("./data")
        
        rag.create_agent()

    rag = st.session_state["agentic"]
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Enter a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.run_agent(prompt)
                st.write_stream(response.response_gen)
                print(response.response)
                st.session_state.messages.append({"role": "agentic", "content": response.response})