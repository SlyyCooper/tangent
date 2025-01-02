<div align="center">
  <h1>🦊 TANGENT</h1>
</div>

<div align="center">
  <img src="public/tangent.webp" alt="Tangent" width="800" height="200" style="object-fit: contain;"/>
</div>




*Inspired by [OpenAI's Swarm](https://github.com/openai/swarm)*

> A lightweight, ergonomic framework for building and orchestrating multi-agent systems. Created by [SlyyCooper](https://github.com/SlyyCooper).

Tangent focuses on making agent **coordination** and **execution** lightweight, highly controllable, and easily testable.

It accomplishes this through three primitive abstractions:
1. `Agent`s (encompassing instructions and tools)
2. **Handoffs** (allowing agents to transfer control)
3. **Triage** (automatic orchestration and routing)

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing you to build scalable, real-world solutions while avoiding a steep learning curve.

> [!NOTE]
> tangent Agents are not related to Assistants in the Assistants API. They are named similarly for convenience, but are otherwise completely unrelated. tangent is entirely powered by the Chat Completions API and is hence stateless between calls.

## Important Notes

- **Model Support**: Uses `gpt-4o` as the default model (GPT-4 is deprecated in this codebase)
- **Embedding Model**: Default is `text-embedding-3-large`
- **Vector Database Support**: 
  - Qdrant (default)
  - Pinecone
  - Custom implementations

## Install

Requires Python 3.10+

```shell
# SSH installation
pip install git+ssh://git@github.com/SlyyCooper/tangent_agents.git

# HTTPS installation
pip install git+https://github.com/SlyyCooper/tangent_agents.git
```

## Quick Start Example

Here's a basic example to show how to use Tangent:

```python
from tangent import tangent, Agent

client = tangent()

def transfer_to_agent_b():
    return agent_b

agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    instructions="Only speak in Haikus.",
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],
)

print(response.messages[-1]["content"])
```

```
Hope glimmers brightly,
New paths converge gracefully,
What can I assist?
```

## Core Concepts

### Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

#### `Agent` Fields

| Field                | Type                     | Description                                                                   | Default                      |
| -------------------- | ------------------------ | ----------------------------------------------------------------------------- | ---------------------------- |
| **name**             | `str`                    | The name of the agent.                                                        | `"Agent"`                    |
| **model**            | `str`                    | The model to be used by the agent.                                            | `"gpt-4o"`                   |
| **instructions**     | `str` or `func() -> str` | Instructions for the agent, can be a string or a callable returning a string. | `"You are a helpful agent."` |
| **functions**        | `List`                   | A list of functions that the agent can call.                                  | `[]`                         |
| **tool_choice**      | `str`                    | The tool choice for the agent, if any.                                        | `None`                       |
| **triage_assignment**| `str`                    | The name of the triage agent this agent is assigned to.                       | `None`                       |

### Functions

- tangent `Agent`s can call python functions directly.
- Function should usually return a `str` (values will be attempted to be cast as a `str`).
- If a function returns an `Agent`, execution will be transferred to that `Agent`.
- If a function defines a `context_variables` parameter, it will be populated by the `context_variables` passed into `client.run()`.

```python
def greet(context_variables, language):
   user_name = context_variables["user_name"]
   greeting = "Hola" if language.lower() == "spanish" else "Hello"
   print(f"{greeting}, {user_name}!")
   return "Done"

agent = Agent(
   functions=[greet]
)

client.run(
   agent=agent,
   messages=[{"role":"user", "content": "Usa greet() por favor."}],
   context_variables={"user_name": "John"}
)
```

```
Hola, John!
```

- If an `Agent` function call has an error (missing function, wrong argument, error) an error response will be appended to the chat so the `Agent` can recover gracefully.
- If multiple functions are called by the `Agent`, they will be executed in that order.

#### Function Schemas

tangent automatically converts functions into a JSON Schema that is passed into Chat Completions `tools`.

- Docstrings are turned into the function `description`.
- Parameters without default values are set to `required`.
- Type hints are mapped to the parameter's `type` (and default to `string`).
- Per-parameter descriptions are not explicitly supported, but should work similarly if just added in the docstring. (In the future docstring argument parsing may be added.)

```python
def greet(name, age: int, location: str = "New York"):
   """Greets the user. Make sure to get their name and age before calling.

   Args:
      name: Name of the user.
      age: Age of the user.
      location: Best place on earth.
   """
   print(f"Hello {name}, glad you're {age} in {location}!")
```

```javascript
{
   "type": "function",
   "function": {
      "name": "greet",
      "description": "Greets the user. Make sure to get their name and age before calling.\n\nArgs:\n   name: Name of the user.\n   age: Age of the user.\n   location: Best place on earth.",
      "parameters": {
         "type": "object",
         "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "location": {"type": "string"}
         },
         "required": ["name", "age"]
      }
   }
}
```

### Handoffs

An `Agent` can hand off to another `Agent` by returning it in a `function`.

```python
sales_agent = Agent(name="Sales Agent")

def transfer_to_sales():
   return sales_agent

agent = Agent(functions=[transfer_to_sales])

response = client.run(agent, [{"role":"user", "content":"Transfer me to sales."}])
print(response.agent.name)
```

```
Sales Agent
```

### Context Variables

It can also update the `context_variables` by returning a more complete `Result` object. This can also contain a `value` and an `agent`, in case you want a single function to return a value, update the agent, and update the context variables (or any subset of the three).

```python
sales_agent = Agent(name="Sales Agent")

def talk_to_sales():
   print("Hello, World!")
   return Result(
       value="Done",
       agent=sales_agent,
       context_variables={"department": "sales"}
   )

agent = Agent(functions=[talk_to_sales])

response = client.run(
   agent=agent,
   messages=[{"role": "user", "content": "Transfer me to sales"}],
   context_variables={"user_name": "John"}
)
print(response.agent.name)
print(response.context_variables)
```

```
Sales Agent
{'department': 'sales', 'user_name': 'John'}
```

> [!NOTE]
> If an `Agent` calls multiple functions to hand-off to an `Agent`, only the last handoff function will be used.

### Streaming

```python
stream = client.run(agent, messages, stream=True)
for chunk in stream:
   print(chunk)
```

Uses the same events as [Chat Completions API streaming](https://platform.openai.com/docs/api-reference/streaming). See `process_and_print_streaming_response` in `/tangent/repl/repl.py` as an example.

Two new event types have been added:

- `{"delim":"start"}` and `{"delim":"end"}`, to signal each time an `Agent` handles a single message (response or function call). This helps identify switches between `Agent`s.
- `{"response": Response}` will return a `Response` object at the end of a stream with the aggregated (complete) response, for convenience.

## Triage Agent

The triage agent is a specialized orchestrator that automatically manages and routes requests to other agents. It provides:

1. **Automatic Agent Discovery**
   - Agents can be assigned to a triage agent using the `triage_assignment` field
   - The triage agent automatically discovers and manages assigned agents

2. **Dynamic Transfer Functions**
   - Transfer functions are automatically created for each managed agent
   - Each managed agent gets a transfer back function to return to the triage agent

3. **Smart Routing**
   - Routes requests to appropriate specialized agents based on their capabilities
   - Maintains conversation context across transfers
   - Can handle requests directly when no specialization is needed

### Creating a Triage Agent

```python
from tangent.triage.agent import create_triage_agent

triage_agent = create_triage_agent(
    name="Orchestrator",
    instructions="Route requests to specialized agents",
    auto_discover=True  # Enable automatic agent discovery
)
```

### Assigning Agents to Triage

```python
specialized_agent = Agent(
    name="Specialist",
    instructions="Handle specialized tasks",
    functions=[special_function],
    triage_assignment="Orchestrator"  # Must match triage agent name
)
```

The triage agent will automatically:
1. Discover the assigned agent
2. Create transfer functions
3. Add transfer back capability
4. Update its instructions with agent information

### Detailed Triage Agent Guide

Welcome to the comprehensive guide for building and running triage (orchestration) agents with Tangent. This section provides an in-depth look at creating, configuring, and running triage agents.

#### Prerequisites

- **Python 3.10+** (Tangent requires 3.10 or higher)
- The **Tangent** library installed
- Basic understanding of agent-based systems

#### Basic Agent Types

Before diving into triage, let's explore the three fundamental agent types:

##### 1. Basic Agent (No Tools)

```python
from tangent import Agent, tangent

client = tangent()

basic_agent = Agent(
    name="BasicAgent",
    instructions="You are a helpful agent that responds politely to user input."
)
```

##### 2. Agent with Tools (Function Calls)

```python
def greet_user(name: str):
    """
    Greet a user by name.
    """
    return f"Hello, {name}! Hope you're having a great day."

tool_agent = Agent(
    name="ToolAgent",
    instructions="You can greet users by name using greet_user().",
    functions=[greet_user]
)
```

##### 3. Agent with Embeddings

```python
from tangent.types import EmbeddingConfig, QdrantConfig
from tangent.embeddings import DocumentStore

# Configure embeddings
embedding_config = EmbeddingConfig(
    model="text-embedding-3-large",
    vector_db=QdrantConfig(
        collection_name="my_collection",
        url="localhost",
        port=6333
    )
)

# Set up document store
doc_store = DocumentStore(
    documents_path="path/to/docs",
    config=embedding_config
)

def search_docs(query: str, top_k: int = 3) -> Result:
    try:
        results = doc_store.search(query, top_k)
        text_results = "\n\n".join(doc.text for doc in results)
        return Result(value=text_results)
    except Exception as e:
        return Result(value=f"Error: {e}")

embedding_agent = Agent(
    name="EmbeddingAgent",
    instructions="You can answer user questions by looking up documents.",
    functions=[search_docs]
)
```

#### Triage Agent Implementation

The triage agent serves as an orchestrator, managing and routing requests to specialized agents. Here's how to set it up:

##### 1. Creating the Triage Agent

```python
from tangent.triage.agent import create_triage_agent

triage_agent = create_triage_agent(
    name="Triage Agent",
    instructions="""You are the orchestrator. Route requests to specialized agents 
    or handle them yourself.""",
    auto_discover=True  # Auto-discovers assigned agents
)
```

##### 2. Creating Specialized Agents

```python
# Sales Agent Example
sales_agent = Agent(
    name="Sales Agent",
    instructions="Handle sales inquiries. Offer products and manage orders.",
    functions=[offer_discount],  # Your sales-specific tools
    triage_assignment="Triage Agent"  # Must match triage agent's name
)

# Document Assistant Example
docs_agent = Agent(
    name="Document Assistant",
    instructions="Search and provide information from our document base.",
    functions=[search_documents],
    triage_assignment="Triage Agent"
)
```

#### How Triage Works

1. **Discovery**: The triage agent automatically discovers agents assigned to it via `triage_assignment`
2. **Transfer Functions**: Creates transfer functions for each discovered agent
3. **Routing**: Analyzes requests and routes to appropriate specialized agents
4. **Context**: Maintains conversation context across transfers

#### Triage Flow Example

```python
# 1. User sends message to triage agent
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "I want to buy something"}]
)

# 2. Triage agent analyzes and transfers to sales agent if appropriate
# 3. Sales agent handles request with its specialized tools
# 4. Sales agent can transfer back when done
```

#### Advanced Features

##### 1. Embedding Integration

For knowledge-based agents:

```python
from tangent.types import Result, QdrantConfig, EmbeddingConfig
from tangent.embeddings import DocumentStore

# Set up document store with configuration
doc_store = DocumentStore(
    documents_path="path/to/docs",
    config=EmbeddingConfig(
        model="text-embedding-3-large",
        vector_db=QdrantConfig(
            collection_name="my_docs",
            url="localhost",
            port=6333
        )
    )
)

def search_documents(query: str, top_k: int = 3) -> Result:
    try:
        results = doc_store.search(query, top_k)
        formatted_results = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.text}"
            for doc in results
        ])
        return Result(value=formatted_results)
    except Exception as e:
        return Result(value=f"Error searching documents: {e}")

# Create knowledge-based agent
docs_agent = Agent(
    name="Document Assistant",
    instructions="You have access to our document knowledge base.",
    functions=[search_documents],
    triage_assignment="Triage Agent"
)
```

## Document Processing

The DocumentStore supports multiple formats:
- `.txt` (plain text)
- `.md` (markdown)
- `.pdf` (PDF documents)
- `.docx` (Word documents)
- `.json` (JSON with content/metadata)

Features:
- Automatic document reading and chunking
- Automatic embedding generation
- Vector database storage (Qdrant/Pinecone)
- Semantic search capabilities

## Advanced Topics

### Multi-Turn Conversations & Context

#### Multi-Turn Basics

Tangent can handle multi-turn conversations within a single `client.run(...)` call. By default, the code in `core.py` loops until there are no more function calls or until `max_turns` is reached.

- **Each time** the model responds, Tangent checks if it wants to call any tools (function calls).
- If so, it executes them and appends the results to the conversation.
- The conversation can continue multiple times until the model stops calling new tools.

#### Passing Context History

Messages from previous user interactions are appended to the `messages` list. You can keep adding new user messages to `messages` to preserve a conversation's history. Tangent automatically includes them in the next call.

**Example**:

```python
messages = []
while True:
    user_input = input("User: ")
    messages.append({"role": "user", "content": user_input})
    
    response = client.run(
        agent=my_agent,
        messages=messages
    )
    
    # The agent's final response:
    print(response.messages[-1]["content"])
    messages.extend(response.messages)  # preserve them for next turn
```

#### Passing Context Between Agents

When an agent **hands off** to another agent (by returning an `Agent` in a function call or using triage-based `transfer`), the conversation's **`context_variables`** also get passed along. If your agent function returns a `Result` with `context_variables`, those updates are merged into the context before the next agent takes control.

**Example**:
```python
def transfer_with_context():
    """
    Transfer to Sales Agent, but also store user preference in context.
    """
    return Result(
        value="Switching you to Sales.",
        agent=sales_agent,
        context_variables={"preferred_color": "blue"}
    )
```

### Timed or Event-Triggered Agents

Tangent itself is stateless between calls, but you can build scheduling or triggers around it:

- **External Cron/Timer**: At a scheduled time, run a function that calls `client.run(...)` with a specific agent (e.g., a "Reminder Agent").
- **Event-Driven**: If an external event or database insert triggers something, you can programmatically call `client.run(...)` with that agent.

**Example** (pseudo-code):

```python
import time

while True:
    time.sleep(60)  # check every minute
    if check_database_for_updates():
        messages = [{"role": "user", "content": "New data arrived in DB, handle it."}]
        response = client.run(agent=db_agent, messages=messages)
        ...
```

### Connecting Agents to a Database

Agents can call **tools** (Python functions) that interact with any database. For instance:

```python
def store_in_db(data: dict):
    """
    Insert data into our MySQL database.
    """
    # your DB logic here...
    return "Data stored!"
```

Attach to an agent:

```python
db_agent = Agent(
    name="Database Agent",
    instructions="You can store data in our DB using store_in_db().",
    functions=[store_in_db]
)
```

### Agent Activation on Specific Conditions

If you want an agent to be "activated" only when certain conditions are met, you have several options:

1. **Triage Logic**: The triage agent can call a specialized agent only if the user's request matches certain instructions.
2. **Model Logic**: The model can interpret requests and call the appropriate function or transfer function automatically.
3. **Manual Logic**: You can filter user requests yourself in Python and decide to call `client.run(..., agent=some_agent)` only if certain text or conditions match.

For instance, you might parse user input for keywords like "urgent" or "report," then decide to run a specialized "Alert Agent."

## Examples

Check out `/examples` for inspiration! Learn more about each one in its README.


- [`triage_agent`](examples/triage_agent): Example of automatic agent discovery and routing using the triage agent. Passing users to the 'websearch_agent'or 'embedding_agent' based on their request.
- [`websearch_agent`](examples/websearch_agent): Example of an agent that can search the web and summarize results
- [`embedding_agent`](examples/embedding_agent): Example of using embeddings for semantic search and document retrieval

## Evaluations

Evaluations are crucial to any project, and we encourage developers to bring their own eval suites to test the performance of their tangents. For reference, we have some examples for how to eval tangent in the `airline`, `weather_agent` and `triage_agent` quickstart examples. See the READMEs for more details.

### Testing and Evaluation

Create test cases in `evals.py`:

```python
import pytest
from tangent import tangent

def test_sales_routing():
    client = tangent()
    response = client.run(
        agent=triage_agent,
        messages=[{"role": "user", "content": "I want to buy shoes"}]
    )
    assert response.agent.name == "Sales Agent"

def test_document_routing():
    client = tangent()
    response = client.run(
        agent=triage_agent,
        messages=[{"role": "user", "content": "What's in our documentation?"}]
    )
    assert response.agent.name == "Document Assistant"
```

Run tests with:
```bash
pytest evals.py
```

## Utils

Use the `run_tangent_loop` to test out your tangent! This will run a REPL on your command line. Supports streaming.

```python
from tangent.repl import run_tangent_loop
...
run_tangent_loop(agent, stream=True)
```

## FAQ / Troubleshooting

1. **How do I preserve multi-turn context across multiple calls?**  
   - Ensure you keep appending the returned messages (from `response.messages`) to your local `messages` list. Then pass that updated `messages` list to the next `client.run(...)`.  

2. **How do I pass context from one agent to another automatically?**  
   - If an agent calls `return Result(agent=other_agent, context_variables={"foo": "bar"})`, the conversation automatically switches to `other_agent` and merges `{"foo": "bar"}` into `context_variables`.

3. **Can I connect multiple agents to an external database?**  
   - Yes. Each agent can have one or more functions that interface with your database. The model can call them automatically via function calling.

### Troubleshooting

1. **Agent Discovery Issues**
   - Verify `triage_assignment` matches triage agent name exactly
   - Ensure `auto_discover=True` is set
   - Check agent imports are in scope for `sys.modules`

2. **Tool Execution Errors**
   - Verify function names match exactly
   - Check function signatures and docstrings
   - Ensure all required parameters are provided

3. **Model Configuration**
   - Use `"gpt-4o"` (default) for best results
   - GPT-4 is deprecated in this codebase