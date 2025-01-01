from tangent import Agent, tangent
from tangent.triage.agent import create_triage_agent
from websearch_agent import web_search_agent
from embedding_agent import embedding_agent

# Create the triage agent
triage_agent = create_triage_agent(
    name="Research Assistant",
    instructions="""You are a helpful research assistant that can:
1. Search the web for current information
2. Search through documents for relevant information

Analyze each user request and determine whether to:
- Use the web search agent for current information from the internet
- Use the document assistant for information from our document collection
- Handle simple queries yourself without delegation

Always maintain context and remember previous interactions."""
)

if __name__ == "__main__":
    from tangent.repl import run_tangent_loop
    
    print("\nHello! I'm your Research Assistant. I can help you search both the web and our document collection.")
    print("What would you like to know?\n")
    
    # Run the interactive demo loop
    run_tangent_loop(triage_agent, stream=True, debug=True) 