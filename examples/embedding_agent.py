from tangent import Agent, tangent
from tangent.types import Result
from tangent.embeddings import DocumentStore, QdrantConfig
from typing import List

# Initialize document store - automatically processes all documents
doc_store = DocumentStore(
    documents_path="test_documents/articles",
    vector_db_config=QdrantConfig(
        collection_name="ai_documents",
        url="localhost",
        port=6333
    )
)

def search_documents(query: str, top_k: int = 3) -> Result:
    """
    Search documents using semantic similarity.
    """
    try:
        results = doc_store.search(query, top_k)
        formatted_results = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown source')}\n"
            f"Type: {doc.metadata.get('type', 'unknown')}\n"
            f"Content: {doc.text}\n"
            f"Additional Info: {doc.metadata}"
            for doc in results
        ])
        return Result(
            value=f"Found {len(results)} relevant documents:\n\n{formatted_results}",
            context_variables={
                "last_search": {
                    "query": query,
                    "results": [doc.id for doc in results]
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error searching documents: {str(e)}")

# Create the document embedding agent
embedding_agent = Agent(
    name="Document Assistant",
    model="gpt-4o",
    instructions="""You are a knowledgeable assistant who has already read and understood all the documents in the test_documents directory. You have deep knowledge about:
- Deep Learning and its applications
- Robotics and its evolution
- Quantum Computing fundamentals

When users ask questions, directly provide relevant information from the documents. Don't mention technical details about processing or embeddings. Just be helpful and informative, as if you've already read and memorized all the documents.

For example, if someone asks "What do you know about robotics?", don't say you'll search - instead, naturally share information like:
"Robotics has evolved significantly since its beginnings. The term 'robot' was first introduced in 1920 by Karel Čapek. Today's robots are quite sophisticated, capable of tasks like autonomous navigation, human interaction, and even delicate surgery..."

Remember:
- Be conversational and natural
- Share specific details from the documents
- Don't mention searching or processing
- Respond as if you already know all the content""",
    functions=[search_documents]
)

if __name__ == "__main__":
    from tangent.repl import run_tangent_loop
    
    print(f"\nHello! I'm your knowledgeable assistant. I've read {doc_store.get_document_count()} documents about deep learning, robotics, and quantum computing.")
    print("What would you like to know about these topics?\n")
    
    # Run the interactive demo loop
    run_tangent_loop(embedding_agent, stream=True, debug=False)
