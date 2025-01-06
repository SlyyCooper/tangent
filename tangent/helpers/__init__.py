from tangent import tangent, Agent, Response
from typing import Union, Optional, List

def setup_agent(
    name: str = "Assistant",
    model: str = "gpt-4o",
    instructions: str = "You are a helpful assistant.",
    vision: bool = False
) -> tuple[tangent, Agent]:
    """
    Helper function to quickly set up a tangent client and agent.
    
    Args:
        name: Name of the agent
        model: Model to use (gpt-4o)
        instructions: Agent instructions
        vision: Whether to enable vision capabilities (default: False)
        
    Returns:
        Tuple of (tangent client, configured agent)
    """
    client = tangent()
    agent = Agent(
        name=name,
        model=model,
        instructions=instructions,
        vision_enabled=vision  # Set vision flag directly
    )
    return client, agent

def get_user_input(prompt: str = "User: ") -> str:
    """
    Helper function to get user input with optional custom prompt.
    
    Args:
        prompt: The prompt to show before user input (default: "User: ")
        
    Returns:
        The user's input as a string
    """
    return input(f"\033[90m{prompt}\033[0m")

def show_ai_response(response, agent_name: str = None):
    """
    Helper function to show AI responses. Handles both streaming and non-streaming responses.
    Defaults to streaming display unless response is non-streaming.
    
    Args:
        response: The response from the agent (streaming or non-streaming)
        agent_name: Optional name to show before responses
    """
    # Handle non-streaming response (Response object)
    if hasattr(response, 'messages'):
        messages = response.messages
    # Handle non-streaming response (dict with messages)
    elif isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
    # Handle streaming response (default)
    else:
        content = ""
        last_sender = agent_name

        for chunk in response:
            if "sender" in chunk:
                last_sender = chunk["sender"]

            if "content" in chunk and chunk["content"] is not None:
                if not content and last_sender:
                    print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                print(chunk["content"], end="", flush=True)
                content += chunk["content"]

            if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                for tool_call in chunk["tool_calls"]:
                    f = tool_call["function"]
                    name = f["name"]
                    if not name:
                        continue
                    print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

            if "delim" in chunk and chunk["delim"] == "end" and content:
                print()  # End of response message
                content = ""
        return

    # Process non-streaming messages
    for message in messages:
        if message["role"] != "assistant":
            continue
        
        # Print agent name in blue
        sender = message.get("sender", agent_name)
        print(f"\033[94m{sender}\033[0m:", end=" ")
        
        # Print response content
        if message.get("content"):
            print(message["content"])
        
        # Print tool calls in purple
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name = f["name"]
            if not name:
                continue
            print(f"\033[94m{sender}: \033[95m{name}\033[0m()")

def process_chat(
    client: 'tangent',
    agent: 'Agent',
    message: str,
    image: Optional[str] = None,
    images: Optional[List[str]] = None,
    stream: bool = True
) -> Union[dict, 'Response']:
    """
    Helper function to process a chat interaction with an AI agent.
    Handles both sending the message and getting the response.
    
    Args:
        client: The tangent client
        agent: The AI agent to chat with
        message: The message to send to the AI
        image: Optional path or URL to a single image
        images: Optional list of image paths or URLs
        stream: Whether to stream the response (default: True)
        
    Returns:
        The AI's response (streaming or complete)
    """
    return client.run(
        agent=agent,
        messages=[{"role": "user", "content": message}],
        image=image,
        images=images,
        stream=stream
    )

# Export the helper functions
__all__ = ["setup_agent", "get_user_input", "show_ai_response", "process_chat"] 