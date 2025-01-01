from typing import List
from ..types import Result, Agent

def search_knowledge_base(query: str, top_k: int = 5, context_variables: dict = None, agent: Agent = None) -> Result:
    """
    Search the agent's knowledge base for relevant documents.
    
    Args:
        query: The search query
        top_k: Number of results to return
        context_variables: Current conversation context
        agent: The agent instance
    """
    if not agent or not agent._embedding_manager:
        return Result(
            value="Knowledge base search is not configured for this agent.",
            context_variables=context_variables or {}
        )
    
    try:
        # Search for relevant documents
        documents = agent.search_knowledge_base(query, top_k)
        
        # Format results
        if not documents:
            return Result(
                value="No relevant documents found.",
                context_variables=context_variables or {}
            )
        
        # Format results nicely
        formatted_results = "\n\n".join([
            f"Document: {doc.metadata.get('title', 'Untitled')}\n"
            f"Content: {doc.text}\n"
            f"Relevance: {doc.metadata.get('score', 'N/A')}"
            for doc in documents
        ])
        
        return Result(
            value=formatted_results,
            context_variables={
                **(context_variables or {}),
                "last_search_query": query,
                "num_results": len(documents)
            }
        )
        
    except Exception as e:
        return Result(
            value=f"Error searching knowledge base: {str(e)}",
            context_variables=context_variables or {}
        ) 