from typing import List
from ..types import Structured_Result, Agent

def search_knowledge_base(query: str, top_k: int = 5, extracted_data: dict = None, agent: Agent = None) -> Structured_Result:
    """
    Search the agent's knowledge base for relevant documents.
    
    Args:
        query: The search query
        top_k: Number of results to return
        extracted_data: Current conversation context
        agent: The agent instance
    """
    if not agent or not agent._embedding_manager:
        return Structured_Result(
            value="Knowledge base search is not configured for this agent.",
            extracted_data=extracted_data or {}
        )
    
    try:
        # Search for relevant documents
        documents = agent.search_knowledge_base(query, top_k)
        
        # Format results
        if not documents:
            return Structured_Result(
                value="No relevant documents found.",
                extracted_data=extracted_data or {}
            )
        
        # Format results nicely
        formatted_results = "\n\n".join([
            f"Document: {doc.metadata.get('title', 'Untitled')}\n"
            f"Content: {doc.text}\n"
            f"Relevance: {doc.metadata.get('score', 'N/A')}"
            for doc in documents
        ])
        
        return Structured_Result(
            value=formatted_results,
            extracted_data={
                **(extracted_data or {}),
                "last_search_query": query,
                "num_results": len(documents)
            }
        )
        
    except Exception as e:
        return Structured_Result(
            value=f"Error searching knowledge base: {str(e)}",
            extracted_data=extracted_data or {}
        ) 