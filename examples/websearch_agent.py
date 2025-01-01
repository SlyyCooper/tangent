from tangent import Agent, tangent
from tangent.types import Result
import os
from tavily import TavilyClient

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> Result:
    """
    Search the web using Tavily API and return relevant results.
    """
    try:
        response = tavily_client.search(query)
        # Format the results in a readable way
        formatted_results = "\n\n".join([
            f"Title: {result.get('title', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}"
            for result in response.get('results', [])[:3]  # Get top 3 results
        ])
        return Result(
            value=formatted_results,
            context_variables={"last_search_query": query}
        )
    except Exception as e:
        return Result(value=f"Error performing web search: {str(e)}")

# Create the web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    model="gpt-4o",
    instructions="""You are a helpful web search agent that can search the internet for information.
When a user asks a question, use the web_search function to find relevant information.
Always analyze the search results and provide a concise, informative response based on the findings.
If the search results are not relevant or if you need more specific information, you can perform another search with a refined query.""",
    functions=[web_search]
)

# Initialize tangent client
client = tangent()

def run_web_search_conversation(query: str, context_variables: dict = None):
    """
    Run a web search conversation with the agent.
    """
    if context_variables is None:
        context_variables = {}
    
    messages = [{"role": "user", "content": query}]
    
    response = client.run(
        agent=web_search_agent,
        messages=messages,
        context_variables=context_variables,
        stream=True,
        debug=True
    )
    
    return response

if __name__ == "__main__":
    from tangent.repl import run_tangent_loop
    # Run the interactive demo loop
    run_tangent_loop(web_search_agent, stream=True, debug=False)