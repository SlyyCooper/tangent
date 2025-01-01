from tangent import Agent, tangent, run_tangent_loop, process_and_print_streaming_response
from tangent.types import Result
import os
from tavily import TavilyClient

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def greet(name: str) -> Result:
    """
    Greet the user by name.
    """
    return Result(
        value=f"Hi {name}! I'm your web search assistant. How can I help you today?",
        context_variables={"user_name": name}
    )

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
When a user asks who they are or what their name is, check context_variables["user_name"] for their name.
Always remember and use their name from context_variables["user_name"] when addressing them.

For web searches, use the web_search function to find relevant information.
Always analyze the search results and provide a concise, informative response based on the findings.
If the search results are not relevant or if you need more specific information, you can perform another search with a refined query.
When asked about previous messages, check the conversation history in messages to provide accurate responses.""",
    functions=[greet, web_search],
    triage_assignment="Research Assistant"  # Assign to Research Assistant triage agent
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
    # Run the interactive demo loop with initial greeting
    print("Please enter your name:")
    name = input().strip()
    # Use the greet function to start the conversation
    context_variables = {}
    messages = [{"role": "user", "content": f"I'm {name}"}]
    response_stream = client.run(
        agent=web_search_agent,
        messages=messages,
        context_variables=context_variables,
        stream=True,
        debug=False
    )
    # Process the streaming response
    final_response = process_and_print_streaming_response(response_stream)
    # Now run the main loop with the updated context
    run_tangent_loop(web_search_agent, context_variables=final_response.context_variables, stream=True, debug=False)