import streamlit as st
import asyncio
from rag import AgenticRAG

rag = AgenticRAG()

if "messages" not in st.session_state: 
    st.session_state.messages = []

for message in st.session_state.messages: 
    with st.chat_message(message['role']): 
        st.markdown(message['content'])
        # Display tool calls if they exist
        if 'tool_calls' in message and message['tool_calls']:
            st.info("Tools used:")
            for tool_call in message['tool_calls']:
                st.code(f"Tool: {tool_call['name']}\nInput: {tool_call['input']}\nOutput: {tool_call['output']}", language="text")

if prompt := st.chat_input('Ask me a question?'): 
    # Display user message
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get and display assistant response
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            try:
                response, tool_calls = rag.query_sync(prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                response = "I apologize, but I encountered an error processing your request."
                tool_calls = []
        
    # Store assistant response with tool calls
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': response,
        'tool_calls': tool_calls
    })
