from tangent import Result, DocumentStore
from tangent.types import EmbeddingConfig, QdrantConfig
import os

# Get the absolute path to test_documents directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs_path = os.path.join(base_dir, "my_documents")

# Initialize document store with configuration
doc_store = DocumentStore(
    documents_path=docs_path,
    config=EmbeddingConfig(
        model="text-embedding-3-large",
        chunk_size=500,
        vector_db=QdrantConfig(
            collection_name="my_documents",
            url="localhost",
            port=6333
        )
    )
)

def search_documents(query: str, top_k: int = 3) -> Result:
    """
    Search documents using semantic similarity.
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 3)
        
    Returns:
        Result object containing formatted search results and context
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
            extracted_data={
                "last_search": {
                    "query": query,
                    "results": [doc.id for doc in results]
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error searching documents: {str(e)}")

def get_document_count() -> int:
    """
    Get the total number of documents in the document store.
    
    Returns:
        int: Number of documents
    """
    return doc_store.get_document_count()
