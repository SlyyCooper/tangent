# 🦊 TANGENT

<div align="center">
  <img src="public/tangent.webp" alt="Tangent" width="600"/>
</div>

A modern, lightweight framework for building AI agents with seamless model switching between OpenAI and Anthropic. Created by [SlyyCooper](https://github.com/SlyyCooper).

## Features

- 🔄 **Unified Model Interface**: Switch between OpenAI and Anthropic models by just changing the model name
- 🛠️ **Rich Helper Functions**: Streamlined setup and interaction with AI agents
- 🔧 **Function Calling**: Seamless tool integration across both providers
- 📚 **Document Management**: Built-in vector storage and semantic search
- 🔀 **Triage System**: Automatic request routing between specialized agents
- 🌊 **Real-time Streaming**: Async streaming support for responsive interactions

## Installation

Requires Python 3.10+

```shell
# SSH installation
pip install git+ssh://git@github.com/SlyyCooper/tangent.git

# HTTPS installation
pip install git+https://github.com/SlyyCooper/tangent.git
```

## Quick Start

```python
from tangent import setup_agent, get_user_input, show_ai_response, process_chat

# Quick setup - works with both OpenAI and Anthropic
client, agent = setup_agent(
    name="ChatBot",
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022"
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
```

## Model Support

### OpenAI Models
- `"gpt-4o"` - Latest GPT-4 model

### Anthropic Models
- `"claude-3-5-sonnet-20241022"` - Latest Claude 3 Sonnet model

Just change the model name and everything works automatically!

## Advanced Usage

### Function-Enabled Agent

```python
from tangent import setup_agent, get_user_input, show_ai_response, process_chat

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Setup with functions
client, agent = setup_agent(
    name="ToolBot",
    model="gpt-4o",
    instructions="You are a helpful assistant that can search the web."
)
agent.functions = [search_web]

# Interactive loop
while True:
    question = get_user_input("Search: ")
    response = process_chat(client, agent, question)
    show_ai_response(response, agent.name)
```

### Knowledge Base Agent

```python
from tangent import setup_agent, get_user_input, show_ai_response, process_chat, DocumentStore

# Set up document store
docs = DocumentStore("my_documents")

# Setup with knowledge base
client, agent = setup_agent(
    name="KnowledgeBot",
    model="gpt-4o",
    instructions="You help answer questions using the knowledge base."
)
agent.embedding_manager = docs.manager

# Interactive loop
while True:
    question = get_user_input("Ask KB: ")
    response = process_chat(client, agent, question)
    show_ai_response(response, agent.name)
```

### Triage System

```python
from tangent import setup_agent, create_triage_agent, process_chat

# Create specialized agents
search_agent = Agent(
    name="SearchBot",
    model="gpt-4o",
    instructions="You handle web searches.",
    triage_assignment="search"
)

kb_agent = Agent(
    name="KnowledgeBot",
    model="gpt-4o",
    instructions="You handle knowledge base queries.",
    triage_assignment="knowledge"
)

# Create triage agent
triage_agent = create_triage_agent(
    name="Router",
    model="gpt-4o",
    instructions="Route requests to appropriate agents.",
    managed_agents=[search_agent, kb_agent]
)

# Use like any other agent
while True:
    question = get_user_input("Ask: ")
    response = process_chat(client, triage_agent, question)
    show_ai_response(response, triage_agent.name)
```

## Core Components

### Agents
- Represent AI assistants with specific capabilities
- Support both OpenAI and Anthropic models
- Can be equipped with functions and tools

### Functions
- Add custom capabilities to agents
- Automatically handled across different models
- Support for parallel execution

### Document Management
- Built-in vector storage support
- Semantic search capabilities
- Support for multiple vector databases:
  - Qdrant (default)
  - Pinecone
  - Custom implementations

### Triage System
- Automatic request routing
- Dynamic agent discovery
- Seamless transfers between agents

## Helper Functions

- `setup_agent()`: Quick agent initialization
- `get_user_input()`: Standardized input handling
- `show_ai_response()`: Universal response display
- `process_chat()`: Streamlined message processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.