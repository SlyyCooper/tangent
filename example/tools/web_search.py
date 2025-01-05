from tangent.types import Result
import os
from tavily import TavilyClient

# Initialize Tavily client with error handling
try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    print(f"Warning: Failed to initialize Tavily client. Make sure TAVILY_API_KEY is set. Error: {e}")
    tavily_client = None

def web_search(query: str, max_results: int = 3) -> Result:
    """
    Search the web using Tavily API and return relevant results.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Result object containing formatted search results and context
    """
    if not tavily_client:
        return Result(value="Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable.")
    
    try:
        response = tavily_client.search(query)
        formatted_results = "\n\n".join([
            f"Title: {result.get('title', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}"
            for result in response.get('results', [])[:max_results]
        ])
        
        return Result(
            value=formatted_results,
            context_variables={
                "last_web_search": {
                    "query": query,
                    "urls": [result.get('url') for result in response.get('results', [])[:max_results]]
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error performing web search: {str(e)}")

def web_extract(urls: list[str]) -> Result:
    """
    Extract raw content from specific URLs using Tavily Extract API.
    
    Args:
        urls: List of URLs to extract content from (max 20 URLs per request)
        
    Returns:
        Result object containing extracted content and metadata
    """
    if not tavily_client:
        return Result(value="Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable.")
    
    if not urls:
        return Result(value="Error: No URLs provided for extraction.")
        
    if len(urls) > 20:
        return Result(value="Error: Maximum of 20 URLs allowed per request.")
    
    try:
        response = tavily_client.extract(urls=urls)
        
        # Format successful extractions
        successful_extracts = []
        for result in response.get('results', []):
            successful_extracts.append(
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {result.get('raw_content', 'N/A')}\n"
            )
            
        # Format failed extractions
        failed_extracts = []
        for failed in response.get('failed_results', []):
            failed_extracts.append(
                f"URL: {failed.get('url', 'N/A')}\n"
                f"Error: {failed.get('error', 'Unknown error')}\n"
            )
            
        # Combine results
        formatted_results = []
        if successful_extracts:
            formatted_results.append("Successfully extracted content:")
            formatted_results.extend(successful_extracts)
        if failed_extracts:
            formatted_results.append("\nFailed extractions:")
            formatted_results.extend(failed_extracts)
            
        return Result(
            value="\n\n".join(formatted_results),
            context_variables={
                "last_web_extract": {
                    "successful_urls": [result.get('url') for result in response.get('results', [])],
                    "failed_urls": [failed.get('url') for failed in response.get('failed_results', [])]
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error performing web extraction: {str(e)}")
