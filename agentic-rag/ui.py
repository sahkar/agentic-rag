import streamlit as st
from rag import AgenticRAG

rag = AgenticRAG()

if "messages" not in st.session_state: 
    st.session_state.messages = []

for message in st.session_state.messages: 
    with st.chat_message(message['role']): 
        st.markdown(message['content'])

if prompt := st.chat_input('Ask me a question?'): 
    # Display user message
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get and display assistant response
    with st.chat_message('assistant'):
        response = rag.query(prompt)
        st.write_stream(response.response_gen)
        
    # Store assistant response
    st.session_state.messages.append({'role': 'assistant', 'content': response})
