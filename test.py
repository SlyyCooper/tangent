from tangent import setup_agent, get_user_input, show_ai_response, process_chat

# Quick setup
client, agent = setup_agent(
    name="ChatBot",
    model="gpt-4o",
    instructions="You are a helpful assistant."
)

# Simple chat loop with streaming
while True:
    # Get user input
    message = get_user_input("You: ")
    
    # Process the chat (streams by default)
    response = process_chat(client, agent, message)
    
    # Show the response
    show_ai_response(response, agent.name)