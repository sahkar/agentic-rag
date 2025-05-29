from graph_rag import GraphRAG
import asyncio

if __name__ == "__main__":
    graph_rag = GraphRAG()
    graph_rag.load_rag()
    graph_rag.create_agent()
    graph_rag.create_context()
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            question = input("Enter a question: ")
            if question == "exit":
                break
            response = loop.run_until_complete(graph_rag.run_agent(question))
            print(response)
    finally:
        loop.close()