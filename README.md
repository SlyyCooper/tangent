# tangent

> tangent is a framework with ergonomic interfaces for multi-agent systems.

## Install

Requires Python 3.10+

```shell
pip install git+ssh://git@github.com/SlyyCooper/tangent_agents.git
```

or

```shell
pip install git+https://github.com/SlyyCooper/tangent_agents.git
```

## Usage

### Basic Agent Example

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

### Triage Agent Example

The triage agent acts as an orchestrator, automatically managing and routing requests to specialized agents:

```python
from tangent import Agent
from tangent.triage.agent import create_triage_agent

# Create specialized agents
web_search_agent = Agent(
    name="Web Search Agent",
    instructions="Search the internet for information",
    functions=[web_search],
    triage_assignment="Research Assistant"  # Assign to triage agent
)

docs_agent = Agent(
    name="Document Assistant",
    instructions="Search through our document collection",
    functions=[search_documents],
    triage_assignment="Research Assistant"
)

# Create the triage agent
triage_agent = create_triage_agent(
    name="Research Assistant",
    instructions="""Route requests to appropriate specialized agents:
    - Web searches -> Web Search Agent
    - Document queries -> Document Assistant"""
)

# The triage agent will automatically:
# 1. Discover assigned agents
# 2. Create transfer functions
# 3. Route requests appropriately
```

Key features of the triage agent:
- Automatic agent discovery via `triage_assignment`
- Dynamic transfer function creation
- Automatic transfer back functionality
- Context preservation across transfers

## Table of Contents

- [Overview](#overview)
- [Examples](#examples)
- [Documentation](#documentation)
  - [Running tangent](#running-tangent)
  - [Agents](#agents)
  - [Functions](#functions)
  - [Streaming](#streaming)
  - [Triage Agent](#triage-agent)
- [Evaluations](#evaluations)
- [Utils](#utils)

# Overview

tangent focuses on making agent **coordination** and **execution** lightweight, highly controllable, and easily testable.

It accomplishes this through three primitive abstractions:
1. `Agent`s (encompassing instructions and tools)
2. **Handoffs** (allowing agents to transfer control)
3. **Triage** (automatic orchestration and routing)

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing you to build scalable, real-world solutions while avoiding a steep learning curve.

> [!NOTE]
> tangent Agents are not related to Assistants in the Assistants API. They are named similarly for convenience, but are otherwise completely unrelated. tangent is entirely powered by the Chat Completions API and is hence stateless between calls.

## Why tangent

tangent explores patterns that are lightweight, scalable, and highly customizable by design. Approaches similar to tangent are best suited for situations dealing with a large number of independent capabilities and instructions that are difficult to encode into a single prompt.

The Assistants API is a great option for developers looking for fully-hosted threads and built in memory management and retrieval. However, tangent is for developers curious to learn about multi-agent orchestration. tangent runs (almost) entirely on the client and, much like the Chat Completions API, does not store state between calls.

# Examples

Check out `/examples` for inspiration! Learn more about each one in its README.

- [`basic`](examples/basic): Simple examples of fundamentals like setup, function calling, handoffs, and context variables
- [`triage_agent`](examples/triage_agent): Example of automatic agent discovery and routing using the triage agent
- [`weather_agent`](examples/weather_agent): Simple example of function calling
- [`airline`](examples/airline): A multi-agent setup for handling different customer service requests in an airline context
- [`support_bot`](examples/support_bot): A customer service bot which includes a user interface agent and a help center agent with several tools
- [`personal_shopper`](examples/personal_shopper): A personal shopping agent that can help with making sales and refunding orders

# Documentation

![tangent Diagram](assets/tangent_diagram.png)

## Running tangent

Start by instantiating a tangent client (which internally just instantiates an `OpenAI` client).

```python
from tangent import tangent

client = tangent()
```

### `client.run()`

tangent's `run()` function is analogous to the `chat.completions.create()` function in the Chat Completions API – it takes `messages` and returns `messages` and saves no state between calls. Importantly, however, it also handles Agent function execution, hand-offs, context variable references, and can take multiple turns before returning to the user.

At its core, tangent's `client.run()` implements the following loop:

1. Get a completion from the current Agent
2. Execute tool calls and append results
3. Switch Agent if necessary
4. Update context variables, if necessary
5. If no new function calls, return

#### Arguments

| Argument              | Type    | Description                                                                                                                                            | Default        |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| **agent**             | `Agent` | The (initial) agent to be called.                                                                                                                      | (required)     |
| **messages**          | `List`  | A list of message objects, identical to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages) | (required)     |
| **context_variables** | `dict`  | A dictionary of additional context variables, available to functions and Agent instructions                                                            | `{}`           |
| **max_turns**         | `int`   | The maximum number of conversational turns allowed                                                                                                     | `float("inf")` |
| **model_override**    | `str`   | An optional string to override the model being used by an Agent                                                                                        | `None`         |
| **execute_tools**     | `bool`  | If `False`, interrupt execution and immediately returns `tool_calls` message when an Agent tries to call a function                                    | `True`         |
| **stream**            | `bool`  | If `True`, enables streaming responses                                                                                                                 | `False`        |
| **debug**             | `bool`  | If `True`, enables debug logging                                                                                                                       | `False`        |

#### `Response` Fields

| Field                 | Type    | Description                                                                                                                                                                                                                                                                  |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **messages**          | `List`  | A list of message objects generated during the conversation. Very similar to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages), but with a `sender` field indicating which `Agent` the message originated from. |
| **agent**             | `Agent` | The last agent to handle a message.                                                                                                                                                                                                                                          |
| **context_variables** | `dict`  | The same as the input variables, plus any changes.                                                                                                                                                                                                                           |

## Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

### `Agent` Fields

| Field                | Type                     | Description                                                                   | Default                      |
| -------------------- | ------------------------ | ----------------------------------------------------------------------------- | ---------------------------- |
| **name**             | `str`                    | The name of the agent.                                                        | `"Agent"`                    |
| **model**            | `str`                    | The model to be used by the agent.                                            | `"gpt-4o"`                   |
| **instructions**     | `str` or `func() -> str` | Instructions for the agent, can be a string or a callable returning a string. | `"You are a helpful agent."` |
| **functions**        | `List`                   | A list of functions that the agent can call.                                  | `[]`                         |
| **tool_choice**      | `str`                    | The tool choice for the agent, if any.                                        | `None`                       |
| **triage_assignment**| `str`                    | The name of the triage agent this agent is assigned to.                       | `None`                       |

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

## Functions

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

### Handoffs and Updating Context Variables

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

### Function Schemas

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
   print(f"Hello {name}, glad you are {age} in {location}!")
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

## Streaming

```python
stream = client.run(agent, messages, stream=True)
for chunk in stream:
   print(chunk)
```

Uses the same events as [Chat Completions API streaming](https://platform.openai.com/docs/api-reference/streaming). See `process_and_print_streaming_response` in `/tangent/repl/repl.py` as an example.

Two new event types have been added:

- `{"delim":"start"}` and `{"delim":"end"}`, to signal each time an `Agent` handles a single message (response or function call). This helps identify switches between `Agent`s.
- `{"response": Response}` will return a `Response` object at the end of a stream with the aggregated (complete) response, for convenience.

# Evaluations

Evaluations are crucial to any project, and we encourage developers to bring their own eval suites to test the performance of their tangents. For reference, we have some examples for how to eval tangent in the `airline`, `weather_agent` and `triage_agent` quickstart examples. See the READMEs for more details.

# Utils

Use the `run_tangent_loop` to test out your tangent! This will run a REPL on your command line. Supports streaming.

```python
from tangent.repl import run_tangent_loop
...
run_tangent_loop(agent, stream=True)
```
