from agentic_rag import AgenticRAG
import asyncio

if __name__ == "__main__":
    agentic_rag = AgenticRAG()
    agentic_rag.load_rag()
    agentic_rag.create_agent()
    agentic_rag.create_context()
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            question = input("Enter a question: ")
            if question == "exit":
                break
            response = loop.run_until_complete(agentic_rag.run_agent(question))
            print(response)
    finally:
        loop.close()
