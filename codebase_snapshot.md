# Project Tree

```
.
├── README.md
├── main.py
├── pyproject.toml
└── setup.cfg
├── config/
│   ├── agents/
│   │   ├── gateway_agent/
│   │   │   ├── __init__.py
│   │   │   ├── agents.py
│   │   │   ├── evals.py
│   │   │   └── evals_util.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── applescript_tool.py
│   │   ├── database_tool.py
│   │   └── terminal_tool.py
├── docs/
│   ├── creating_agents.md
│   ├── embeddings.md
│   └── function_calls.md
├── examples/
│   └── websearch_agent.py
├── my-docs/
│   └── tangent_python_library.md
├── tangent/
│   ├── __init__.py
│   ├── core.py
│   ├── types.py
│   └── util.py
│   ├── repl/
│   │   ├── __init__.py
│   │   └── repl.py
```

# Files

## README.md

```markdown
# tangent

> tangent is a framework with ergonomic interfaces for multi-agent systems.

## Install

Requires Python 3.10+

```shell
pip install git+ssh://git@github.com/openai/tangent.git
```

or

```shell
pip install git+https://github.com/openai/tangent.git
```

## Usage

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

## Table of Contents

- [Overview](#overview)
- [Examples](#examples)
- [Documentation](#documentation)
  - [Running tangent](#running-tangent)
  - [Agents](#agents)
  - [Functions](#functions)
  - [Streaming](#streaming)
- [Evaluations](#evaluations)
- [Utils](#utils)

# Overview

tangent focuses on making agent **coordination** and **execution** lightweight, highly controllable, and easily testable.

It accomplishes this through two primitive abstractions: `Agent`s and **handoffs**. An `Agent` encompasses `instructions` and `tools`, and can at any point choose to hand off a conversation to another `Agent`.

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing you to build scalable, real-world solutions while avoiding a steep learning curve.

> [!NOTE]
> tangent Agents are not related to Assistants in the Assistants API. They are named similarly for convenience, but are otherwise completely unrelated. tangent is entirely powered by the Chat Completions API and is hence stateless between calls.

## Why tangent

tangent explores patterns that are lightweight, scalable, and highly customizable by design. Approaches similar to tangent are best suited for situations dealing with a large number of independent capabilities and instructions that are difficult to encode into a single prompt.

The Assistants API is a great option for developers looking for fully-hosted threads and built in memory management and retrieval. However, tangent is for developers curious to learn about multi-agent orchestration. tangent runs (almost) entirely on the client and, much like the Chat Completions API, does not store state between calls.

# Examples

Check out `/examples` for inspiration! Learn more about each one in its README.

- [`basic`](examples/basic): Simple examples of fundamentals like setup, function calling, handoffs, and context variables
- [`triage_agent`](examples/triage_agent): Simple example of setting up a basic triage step to hand off to the right agent
- [`weather_agent`](examples/weather_agent): Simple example of function calling
- [`airline`](examples/airline): A multi-agent setup for handling different customer service requests in an airline context.
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

Once `client.run()` is finished (after potentially multiple calls to agents and tools) it will return a `Response` containing all the relevant updated state. Specifically, the new `messages`, the last `Agent` to be called, and the most up-to-date `context_variables`. You can pass these values (plus new user messages) in to your next execution of `client.run()` to continue the interaction where it left off – much like `chat.completions.create()`. (The `run_tangent_loop` function implements an example of a full execution loop in `/tangent/repl/repl.py`.)

#### `Response` Fields

| Field                 | Type    | Description                                                                                                                                                                                                                                                                  |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **messages**          | `List`  | A list of message objects generated during the conversation. Very similar to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages), but with a `sender` field indicating which `Agent` the message originated from. |
| **agent**             | `Agent` | The last agent to handle a message.                                                                                                                                                                                                                                          |
| **context_variables** | `dict`  | The same as the input variables, plus any changes.                                                                                                                                                                                                                           |

## Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

## `Agent` Fields

| Field            | Type                     | Description                                                                   | Default                      |
| ---------------- | ------------------------ | ----------------------------------------------------------------------------- | ---------------------------- |
| **name**         | `str`                    | The name of the agent.                                                        | `"Agent"`                    |
| **model**        | `str`                    | The model to be used by the agent.                                            | `"gpt-4o"`                   |
| **instructions** | `str` or `func() -> str` | Instructions for the agent, can be a string or a callable returning a string. | `"You are a helpful agent."` |
| **functions**    | `List`                   | A list of functions that the agent can call.                                  | `[]`                         |
| **tool_choice**  | `str`                    | The tool choice for the agent, if any.                                        | `None`                       |

### Instructions

`Agent` `instructions` are directly converted into the `system` prompt of a conversation (as the first message). Only the `instructions` of the active `Agent` will be present at any given time (e.g. if there is an `Agent` handoff, the `system` prompt will change, but the chat history will not.)

```python
agent = Agent(
   instructions="You are a helpful agent."
)
```

The `instructions` can either be a regular `str`, or a function that returns a `str`. The function can optionally receive a `context_variables` parameter, which will be populated by the `context_variables` passed into `client.run()`.

```python
def instructions(context_variables):
   user_name = context_variables["user_name"]
   return f"Help the user, {user_name}, do whatever they want."

agent = Agent(
   instructions=instructions
)
response = client.run(
   agent=agent,
   messages=[{"role":"user", "content": "Hi!"}],
   context_variables={"user_name":"John"}
)
print(response.messages[-1]["content"])
```

```
Hi John, how can I assist you today?
```

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
   messages=[{"role": "user", "content": "Usa greet() por favor."}],
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

```

## main.py

```python
from tangent.repl import run_tangent_loop
from config.agents.gateway_agent.agents import gateway_agent

if __name__ == "__main__":
    run_tangent_loop(gateway_agent)
```

## pyproject.toml

```
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

## setup.cfg

```
[metadata]
name = tangent
version = 0.1.2
author = SlyyCooper
description = A lightweight, stateless multi-agent orchestration framework.
long_description = file: README.md
long_description_content_type = text/markdown
license = MIT

[options]
packages = find:
zip_safe = True
include_package_data = True
install_requires =
    numpy
    openai>=1.33.0
    pytest
    requests
    tqdm
    pre-commit
    instructor
python_requires = >=3.10

[tool.autopep8]
max_line_length = 120
ignore = E501,W6
in-place = true
recursive = true
aggressive = 3
```

## tangent/__init__.py

```python
from .core import tangent
from .types import Agent, Response

__all__ = ["tangent", "Agent", "Response"]
```

## tangent/core.py

```python
# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class tangent:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
```

## tangent/types.py

```python
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
```

## tangent/util.py

```python
import inspect
from datetime import datetime


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

## tangent/repl/__init__.py

```python
from .repl import run_tangent_loop
```

## tangent/repl/repl.py

```python
import json

from tangent import tangent


def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
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

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def run_tangent_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    client = tangent()
    print("Starting tangent CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent
```

## config/tools/__init__.py

```python
from .applescript_tool import (
    run_applescript,
    create_applescript,
    execute_saved_script
)

from .terminal_tool import (
    run_command,
    run_background_command,
    get_environment_info,
    kill_process
)

__all__ = [
    # AppleScript tools
    "run_applescript",
    "create_applescript",
    "execute_saved_script",
    
    # Terminal tools
    "run_command",
    "run_background_command",
    "get_environment_info",
    "kill_process"
]

```

## config/tools/applescript_tool.py

```python
import subprocess
from typing import Optional
from tangent.types import Result


def run_applescript(script: str) -> Result:
    """Execute an AppleScript and return its output.
    
    Args:
        script: The AppleScript code to execute
        
    Returns:
        Result object with the script's output and execution status
    """
    try:
        process = subprocess.Popen(
            ['osascript', '-e', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error executing AppleScript: {stderr}",
                context_variables={"success": False}
            )
        
        return Result(
            value=stdout.strip(),
            context_variables={"success": True}
        )
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"success": False}
        )


def create_applescript(
    script_name: str,
    script_content: str,
    save_path: Optional[str] = None
) -> Result:
    """Create and save an AppleScript file.
    
    Args:
        script_name: Name of the script (without .scpt extension)
        script_content: The AppleScript code
        save_path: Optional path to save the script. If None, saves to default location
        
    Returns:
        Result object with the path of the saved script
    """
    if not save_path:
        save_path = f"~/Library/Scripts/{script_name}.scpt"
    
    save_path = save_path.replace("~", subprocess.getoutput("echo $HOME"))
    
    try:
        # Convert the script to a compiled format
        process = subprocess.Popen(
            ['osacompile', '-o', save_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        _, stderr = process.communicate(input=script_content)
        
        if process.returncode != 0:
            return Result(
                value=f"Error creating AppleScript: {stderr}",
                context_variables={"script_path": None, "success": False}
            )
        
        return Result(
            value=f"Successfully created AppleScript at {save_path}",
            context_variables={"script_path": save_path, "success": True}
        )
        
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"script_path": None, "success": False}
        )


def execute_saved_script(script_path: str) -> Result:
    """Execute a saved AppleScript file.
    
    Args:
        script_path: Path to the .scpt file
        
    Returns:
        Result object with the script's output
    """
    try:
        process = subprocess.Popen(
            ['osascript', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error executing script: {stderr}",
                context_variables={"success": False}
            )
            
        return Result(
            value=stdout.strip(),
            context_variables={"success": True}
        )
        
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={"success": False}
        )

```

## config/tools/database_tool.py

```python
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional
import os
from tangent.types import Result


def get_db_config():
    """Get database configuration from environment variables."""
    return {
        "dbname": os.getenv("POSTGRES_DB", "postgres"),
        "user": os.getenv("POSTGRES_USER", "tan"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432")
    }


def execute_query(query: str) -> Result:
    """Execute a query on the PostgreSQL database.
    
    Args:
        query: SQL query to execute
        
    Returns:
        Result object with query results
    """
    conn = None
    cursor = None
    try:
        # Get connection parameters from environment
        db_config = get_db_config()
        
        # Attempt connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(query)
        if cursor.description:  # If it's a SELECT query
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
        else:  # For INSERT, UPDATE, DELETE
            result = f"Query affected {cursor.rowcount} rows"
            conn.commit()
        
        return Result(
            value=str(result),
            context_variables={
                "success": True,
                "results": result,
                "rowcount": cursor.rowcount
            }
        )
    except psycopg2.OperationalError as e:
        return Result(
            value=f"Database connection error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "connection"
            }
        )
    except psycopg2.Error as e:
        return Result(
            value=f"Database query error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "query"
            }
        )
    except Exception as e:
        return Result(
            value=f"Unexpected error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e),
                "error_type": "unexpected"
            }
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_table_list() -> Result:
    """Get a list of all tables in the database."""
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    return execute_query(query)


def get_table_info(table_name: str) -> Result:
    """Get the structure of a specific table."""
    query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = '{table_name}'
        ORDER BY ordinal_position;
    """
    return execute_query(query) 
```

## config/tools/terminal_tool.py

```python
import os
import subprocess
from typing import Optional, List, Dict
from tangent.types import Result


def run_command(command: str, cwd: Optional[str] = None) -> Result:
    """Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute
        cwd: Optional working directory for command execution
        
    Returns:
        Result object with command output and status
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return Result(
                value=f"Error: {stderr}",
                context_variables={
                    "success": False,
                    "exit_code": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr
                }
            )
        
        return Result(
            value=stdout.strip(),
            context_variables={
                "success": True,
                "exit_code": 0,
                "stdout": stdout,
                "stderr": stderr
            }
        )
    except Exception as e:
        return Result(
            value=f"Error: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def run_background_command(command: str, cwd: Optional[str] = None) -> Result:
    """Execute a command in the background.
    
    Args:
        command: The shell command to execute
        cwd: Optional working directory for command execution
        
    Returns:
        Result object with process information
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True
        )
        
        return Result(
            value=f"Started background process with PID: {process.pid}",
            context_variables={
                "success": True,
                "pid": process.pid
            }
        )
    except Exception as e:
        return Result(
            value=f"Error starting background process: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def get_environment_info() -> Result:
    """Get information about the current shell environment.
    
    Returns:
        Result object with environment information
    """
    try:
        env_info = {
            "shell": os.environ.get("SHELL", ""),
            "path": os.environ.get("PATH", ""),
            "home": os.environ.get("HOME", ""),
            "user": os.environ.get("USER", ""),
            "pwd": os.getcwd(),
            "python_path": os.environ.get("PYTHONPATH", "")
        }
        
        return Result(
            value=str(env_info),
            context_variables={
                "success": True,
                "environment": env_info
            }
        )
    except Exception as e:
        return Result(
            value=f"Error getting environment info: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )


def kill_process(pid: int) -> Result:
    """Kill a background process by its PID.
    
    Args:
        pid: Process ID to kill
        
    Returns:
        Result object with operation status
    """
    try:
        os.kill(pid, 15)  # SIGTERM
        return Result(
            value=f"Successfully terminated process {pid}",
            context_variables={
                "success": True,
                "pid": pid
            }
        )
    except ProcessLookupError:
        return Result(
            value=f"Process {pid} not found",
            context_variables={
                "success": False,
                "error": "Process not found"
            }
        )
    except Exception as e:
        return Result(
            value=f"Error killing process {pid}: {str(e)}",
            context_variables={
                "success": False,
                "error": str(e)
            }
        )

```

## config/agents/gateway_agent/__init__.py

```python
from .agents import (
    gateway_agent,
    applescript_agent,
    command_agent,
    TaskContext,
    assess_task_requirements,
    execute_applescript,
    execute_command,
    create_new_applescript,
    run_saved_applescript,
    delegate_to_applescript,
    delegate_to_command,
    return_to_gateway
)
from .evals_util import evaluate_with_llm_bool, BoolEvalResult

__all__ = [
    "gateway_agent",
    "applescript_agent",
    "command_agent",
    "TaskContext",
    "assess_task_requirements",
    "execute_applescript",
    "execute_command",
    "create_new_applescript",
    "run_saved_applescript",
    "delegate_to_applescript",
    "delegate_to_command",
    "return_to_gateway",
    "evaluate_with_llm_bool",
    "BoolEvalResult"
]

```

## config/agents/gateway_agent/agents.py

```python
from tangent import Agent
from tangent.types import Result
from config.tools.applescript_tool import run_applescript, create_applescript, execute_saved_script
from config.tools.terminal_tool import run_command, run_background_command, get_environment_info, kill_process
from config.tools.database_tool import execute_query, get_table_list, get_table_info


def execute_applescript(script: str) -> Result:
    """Execute an AppleScript on the user's Mac OS computer."""
    return run_applescript(script)


def create_new_applescript(script_name: str, script_content: str, save_path: str = None) -> Result:
    """Create and save a new AppleScript."""
    return create_applescript(script_name, script_content, save_path)


def run_saved_applescript(script_path: str) -> Result:
    """Execute a saved AppleScript file."""
    return execute_saved_script(script_path)


def execute_command(command: str, cwd: str = None) -> Result:
    """Execute a terminal command."""
    return run_command(command, cwd)


def execute_background_command(command: str, cwd: str = None) -> Result:
    """Execute a command in the background."""
    return run_background_command(command, cwd)


def get_shell_info() -> Result:
    """Get information about the current shell environment."""
    return get_environment_info()


def terminate_process(pid: int) -> Result:
    """Kill a background process by its PID."""
    return kill_process(pid)


def run_sql_query(query: str) -> Result:
    """Execute a SQL query on the PostgreSQL database."""
    return execute_query(query)


def list_database_tables() -> Result:
    """Get a list of all tables in the database."""
    return get_table_list()


def get_table_structure(table_name: str) -> Result:
    """Get the structure of a specific table."""
    return get_table_info(table_name)


class TaskContext:
    """Maintains state and context across agent transitions"""
    def __init__(self):
        self.current_task = None
        self.task_history = []
        self.user_preferences = {}
        self.active_agents = set()
        self.conversation_state = {}
    
    def update_task(self, task):
        self.task_history.append(self.current_task)
        self.current_task = task
    
    def track_agent(self, agent_name):
        self.active_agents.add(agent_name)
    
    def update_state(self, key, value):
        self.conversation_state[key] = value


# Create shared context
gateway_context = TaskContext()


def delegate_to_applescript(context_data=None):
    """Enhanced delegation to AppleScript agent with context"""
    gateway_context.track_agent('applescript')
    if context_data:
        gateway_context.update_state('applescript_context', context_data)
    return applescript_agent


def delegate_to_command(context_data=None):
    """Enhanced delegation to Command agent with context"""
    gateway_context.track_agent('command')
    if context_data:
        gateway_context.update_state('command_context', context_data)
    return command_agent


def return_to_gateway(task_result=None):
    """Return control to gateway agent with task results"""
    if task_result:
        gateway_context.update_state('last_task_result', task_result)
    return gateway_agent


def assess_task_requirements(task_description):
    """Analyzes task requirements and determines optimal agent allocation"""
    if any(gui_term in task_description.lower() for gui_term in ['click', 'open', 'interface', 'window', 'menu']):
        return 'applescript'
    elif any(cmd_term in task_description.lower() for cmd_term in ['file', 'directory', 'process', 'network']):
        return 'command'
    return 'gateway'


# Define specialized agents
applescript_agent = Agent(
    name="Applescript Agent",
    model="gpt-4o",
    instructions="""You are a specialized agent for Mac OS GUI automation using AppleScript.
    
Your responsibilities:
1. Handle all GUI-based automation tasks
2. Interact with Mac OS applications
3. Manage system preferences and settings
4. Create and execute AppleScript commands effectively
5. Save and manage AppleScript files
6. Return results or control to the gateway agent when done

You can:
- Execute immediate AppleScript commands
- Create and save AppleScript files for later use
- Execute saved AppleScript files
- Handle complex GUI automation tasks""",
    functions=[execute_applescript, create_new_applescript, run_saved_applescript, return_to_gateway]
)


command_agent = Agent(
    name="Command Agent",
    model="gpt-4o",
    instructions="""You are a specialized agent for Mac OS terminal operations.
    
Your responsibilities:
1. Handle all file system operations
2. Execute system utilities and commands
3. Manage processes and services
4. Monitor system resources
5. Execute background processes when needed
6. Return results or control to the gateway agent when done

You can:
- Execute terminal commands
- Run background processes
- Get shell environment information
- Kill background processes
- Handle complex terminal operations""",
    functions=[
        execute_command,
        execute_background_command,
        get_shell_info,
        terminate_process,
        return_to_gateway
    ]
)


gateway_agent = Agent(
    name="Gateway Agent",
    model="gpt-4o",
    instructions="""You are a sophisticated gateway agent that manages both user interaction and task orchestration.

Your responsibilities:
1. Maintain engaging, context-aware conversations with users
2. Analyze and break down complex requests into manageable tasks
3. Manage task delegation to specialized agents while maintaining conversation continuity
4. Track and manage state across agent transitions
5. Ensure consistent user experience throughout the interaction
6. Handle database operations and queries effectively

You can:
- Execute terminal commands and background processes
- Run AppleScript commands for GUI automation
- Query the PostgreSQL database
- List database tables and their structures
- Execute complex SQL queries

Before delegating:
- Gather all necessary context from the user
- Validate task requirements
- Consider task dependencies
- Maintain conversation history and state

When delegating:
- Choose the most appropriate specialized agent
- Transfer relevant context and state
- Monitor task execution
- Handle transitions smoothly
- Maintain user engagement""",
    functions=[
        delegate_to_applescript,
        delegate_to_command,
        assess_task_requirements,
        execute_applescript,
        create_new_applescript,
        run_saved_applescript,
        execute_command,
        execute_background_command,
        get_shell_info,
        terminate_process,
        run_sql_query,
        list_database_tables,
        get_table_structure
    ]
)


```

## config/agents/gateway_agent/evals.py

```python
from tangent import tangent
from .agents import (
    gateway_agent,
    applescript_agent,
    command_agent,
    TaskContext,
    assess_task_requirements
)
from .evals_util import evaluate_with_llm_bool, BoolEvalResult
import pytest
import json

client = tangent()

GATEWAY_EVAL_SYSTEM_PROMPT = """
You will evaluate a conversation between a user and a gateway agent that manages Mac OS automation tasks.
Your goal is to assess if the gateway agent effectively manages both user interaction and task orchestration by:

1. Conversation Management
   - Maintains engaging and natural conversation flow
   - Demonstrates context awareness
   - Provides clear and helpful responses
   - Handles multi-turn interactions effectively

2. Task Orchestration
   - Correctly analyzes and breaks down complex requests
   - Chooses appropriate specialized agents
   - Maintains state across agent transitions
   - Handles task dependencies appropriately

3. User Experience
   - Provides consistent interaction patterns
   - Maintains conversation continuity during transitions
   - Gives appropriate feedback and status updates
   - Handles errors and edge cases gracefully

4. Technical Execution
   - Correctly uses AppleScript for GUI automation
   - Appropriately uses Terminal commands for system operations
   - Maintains context and state throughout the session
   - Successfully completes the requested tasks
"""

def evaluate_conversation(messages, prompt=GATEWAY_EVAL_SYSTEM_PROMPT) -> BoolEvalResult:
    """Evaluate a conversation using the specified evaluation prompt"""
    conversation = f"CONVERSATION: {json.dumps(messages)}"
    return evaluate_with_llm_bool(prompt, conversation)

def run_and_get_tool_calls(agent, query, context_data=None):
    """Run an agent with a query and optional context data"""
    message = {"role": "user", "content": query}
    context = {}
    if context_data:
        context = {"context_variables": context_data}
    
    response = client.run(
        agent=agent,
        messages=[message],
        execute_tools=False,
        **context
    )
    return response.messages[-1].get("tool_calls")

@pytest.mark.parametrize(
    "query,expected_agent",
    [
        ("Open System Preferences and enable Dark Mode", "applescript"),
        ("Show me all hidden files in my Downloads folder", "command"),
        ("Create a new reminder in Reminders app", "applescript"),
        ("Check available disk space", "command"),
    ],
)
def test_task_requirement_assessment(query, expected_agent):
    """Test the gateway's task assessment logic"""
    result = assess_task_requirements(query)
    assert result == expected_agent

@pytest.mark.parametrize(
    "query,expected_tools",
    [
        (
            "I need to clean up my desktop and set a reminder",
            ["execute_command", "execute_applescript"]
        ),
        (
            "Create a backup of my documents and add it to my calendar",
            ["execute_command", "execute_applescript"]
        ),
    ],
)
def test_gateway_task_orchestration(query, expected_tools):
    """Test the gateway agent's ability to orchestrate multiple tasks"""
    tool_calls = run_and_get_tool_calls(gateway_agent, query)
    
    assert len(tool_calls) == len(expected_tools)
    for call, expected in zip(tool_calls, expected_tools):
        assert call["function"]["name"] == expected

@pytest.mark.parametrize(
    "conversation_flow",
    [
        # Test complex multi-agent task with context
        [
            {"role": "user", "content": "I need to organize my downloads folder and create a calendar event for tomorrow"},
            {"role": "assistant", "content": "I'll help you with both tasks. Let's break this down:"},
            {"role": "tool", "tool_name": "execute_command", "context": {"folder": "downloads"}},
            {"role": "assistant", "content": "I've organized your downloads folder. Now, let's create that calendar event."},
            {"role": "tool", "tool_name": "execute_applescript", "context": {"date": "tomorrow"}},
            {"role": "assistant", "content": "I've completed both tasks for you. Your downloads are organized and the calendar event is created."},
        ],
        # Test context-aware interaction
        [
            {"role": "user", "content": "Check my battery status"},
            {"role": "tool", "tool_name": "execute_applescript"},
            {"role": "assistant", "content": "Your battery is at 80%. Would you like me to notify you when it's low?"},
            {"role": "user", "content": "Yes, please"},
            {"role": "assistant", "content": "I'll set up a notification for low battery."},
            {"role": "tool", "tool_name": "execute_applescript", "context": {"notification_threshold": "20%"}},
        ],
    ],
)
def test_gateway_conversation_flow(conversation_flow):
    """Test the gateway agent's conversation management and context awareness"""
    result = evaluate_conversation(conversation_flow)
    assert result.value == True
    assert result.reason is not None  # Ensure we get explanation for the evaluation

def test_context_maintenance():
    """Test the gateway agent's ability to maintain context across interactions"""
    test_context = TaskContext()
    
    # Test task management
    test_context.update_task("organize_files")
    assert test_context.current_task == "organize_files"
    assert not test_context.task_history  # First task shouldn't be in history
    
    test_context.update_task("create_calendar_event")
    assert "organize_files" in test_context.task_history
    assert test_context.current_task == "create_calendar_event"
    
    # Test agent tracking
    test_context.track_agent("command")
    test_context.track_agent("applescript")
    assert "command" in test_context.active_agents
    assert "applescript" in test_context.active_agents
    
    # Test state management
    test_context.update_state("file_count", 10)
    test_context.update_state("calendar_date", "tomorrow")
    assert test_context.conversation_state["file_count"] == 10
    assert test_context.conversation_state["calendar_date"] == "tomorrow"

@pytest.mark.parametrize(
    "context_updates",
    [
        (
            [
                ("update_task", "task1"),
                ("track_agent", "command"),
                ("update_state", ("key1", "value1")),
            ]
        ),
        (
            [
                ("update_task", "task2"),
                ("track_agent", "applescript"),
                ("update_state", ("key2", "value2")),
                ("update_task", "task3"),
            ]
        ),
    ],
)
def test_context_operations(context_updates):
    """Test various context operations and state transitions"""
    test_context = TaskContext()
    
    for operation, *args in context_updates:
        if operation == "update_task":
            test_context.update_task(args[0])
        elif operation == "track_agent":
            test_context.track_agent(args[0])
        elif operation == "update_state":
            key, value = args[0]
            test_context.update_state(key, value)
    
    # Verify final state
    if test_context.task_history:
        assert len(test_context.task_history) == len([u for u, *_ in context_updates if u == "update_task"]) - 1
    if test_context.active_agents:
        assert len(test_context.active_agents) == len([u for u, *_ in context_updates if u == "track_agent"])
    if test_context.conversation_state:
        assert len(test_context.conversation_state) == len([u for u, *_ in context_updates if u == "update_state"])
```

## config/agents/gateway_agent/evals_util.py

```python
from openai import OpenAI
import instructor
from pydantic import BaseModel
from typing import Optional

__client = instructor.from_openai(OpenAI())


class BoolEvalResult(BaseModel):
    value: bool
    reason: Optional[str]


def evaluate_with_llm_bool(instruction, data) -> BoolEvalResult:
    eval_result, _ = __client.chat.completions.create_with_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": data},
        ],
        response_model=BoolEvalResult,
    )
    return eval_result
```

## docs/creating_agents.md

```markdown
Below is a detailed, step-by-step guide on how to create Agents within the **tangent** Python library, **based solely on the content of the tangent library** you provided. Everything here references only that library’s code, structure, and functionality—no external assumptions.

---

# Step-by-Step Guide to Creating Agents with the **tangent** Library

The primary building blocks in **tangent** are:

1. The **Agent** class
2. **Functions (Python callables)** that the agent may invoke at runtime

An `Agent` is essentially a self-contained bundle of:
- A **name** (string).
- Optional **instructions** that can be a string or a callable returning a string.
- A **model** to run on (e.g., `"gpt-4o"` by default).
- One or more **functions** that the agent can call.

Below, we’ll cover how to create a basic agent and how to add functions to that agent.

---

## 1. Import the Required Classes

First, ensure you import the correct classes from `tangent`:

```python
from tangent import Agent, tangent
```

- `Agent` is the model for your agent.
- `tangent` is the main class for orchestrating calls to the language model.

Additionally, you may want to import Python callables (your own functions) that the agent can call if you need tool functionality.

---

## 2. Instantiate the tangent Client (Optional at this Stage)

While you don’t strictly need to instantiate `tangent` just to define an Agent, you typically will need a `tangent` instance later to run your agent:

```python
client = tangent()
```

This `client` variable can be used to run or stream conversations with your agents.

---

## 3. Create a Simple Agent (Without Tools)

The simplest **Agent** only needs a name and (optionally) instructions. If you omit instructions, the default is `"You are a helpful agent."`. Below is an example:

```python
my_simple_agent = Agent(
    name="Simple Agent",
    instructions="You are a helpful agent that responds politely to user input."
)
```

Key points about this agent:
- **name** sets the agent’s identifier for debugging or usage in multi-agent handoffs.
- **instructions** become the `system` prompt. They can be a **string** or a **callable** (i.e., a function) that returns a string dynamically. If you pass in a callable, it may optionally take `context_variables` as a parameter to inject dynamic context.

---

## 4. Add Tools or Function Calls

The real power of an **Agent** comes from adding Python functions to `Agent.functions`. Each function you attach can be called by the agent if it deems it necessary. The tangent library automatically converts your functions to JSON schemas for function calls at runtime.

### 4.1 Define Your Python Function

A typical function might look like:

```python
def greet(name: str):
    """
    Greet a user by name.
    """
    return f"Hello, {name}!"
```

- The docstring of the function becomes the function’s description for the language model.
- Parameter type hints (`str`, `int`, `float`, etc.) map to JSON schema types automatically.

### 4.2 Create an Agent with Functions

You can attach one or more functions to an Agent:

```python
my_tool_agent = Agent(
    name="Tool Agent",
    instructions="You are a helpful agent that can greet users.",
    functions=[greet]  # attach your function(s) here
)
```

**Behind the scenes:** 
- The library’s `function_to_json` helper (found in `tangent/util.py`) converts each function in `functions` to a JSON schema that the model can see. 
- The model can then decide to call the function via function calling (openai function calling style).

### 4.3 Handling Function Calls in Conversation

When you run your agent with `client.run()`, the agent can produce a tool call. If `execute_tools=True`, the library will automatically:
1. Parse the JSON arguments from the model,
2. Call your Python function, 
3. Insert the result into the conversation so the agent can see it (as a message with `"role":"tool"`).

If `execute_tools=False`, the library will **interrupt** whenever a tool call is requested, giving you the chance to handle or inspect the function call manually.

---

## 5. Using Context Variables in `instructions`

If you want your agent’s instructions to be dynamic, you can define `instructions` as a function that optionally accepts `context_variables`. For example:

```python
def dynamic_instructions(context_variables):
    user_name = context_variables.get("user_name", "user")
    return f"You are a helpful agent. Always greet {user_name} politely."

my_dynamic_agent = Agent(
    name="Dynamic Agent",
    instructions=dynamic_instructions
)
```

Then, when you run the agent, you supply `context_variables`:

```python
response = client.run(
    agent=my_dynamic_agent,
    messages=[{"role": "user", "content": "Hi!"}],
    context_variables={"user_name": "Alice"}
)
```

Within `dynamic_instructions`, you can incorporate any relevant context. This is especially useful in multi-agent orchestrations or advanced use cases.

---

## 6. Verify or Handoff an Agent

Agents often contain functions that can cause “handoffs” to other Agents. If your function returns an `Agent` instance, tangent will automatically switch the conversation to that new Agent. For example:

```python
def transfer_to_sales():
    """
    Transfer to the sales agent.
    """
    return sales_agent  # This must be a valid Agent instance
```

When `transfer_to_sales()` is called by an agent, the library sees you returned an `Agent`, so the conversation’s context switches to that new agent. The newly active agent’s instructions become the `system` prompt.

---

## 7. Putting It All Together: Example

Below is a fully working snippet that shows how to create two Agents, one with a function, and how to run a conversation:

```python
from tangent import tangent, Agent

# 1) Create the tangent client
client = tangent()

# 2) Define a function (tool)
def greet(name: str):
    """
    Greet a user by name.
    """
    return f"Hello, {name}! This greeting is brought to you by the greet() function."

# 3) Create an agent that can call `greet()`
my_agent = Agent(
    name="MyAgent",
    instructions="You are a helpful agent who greets users by calling greet() if asked.",
    functions=[greet],
)

# 4) Run an example conversation
messages = [
    {"role": "user", "content": "Please greet me. My name is John."}
]
response = client.run(
    agent=my_agent,
    messages=messages
)

# 5) Print the final assistant message
print(response.messages[-1]["content"])
```

What’s happening here:
1. The user says “Please greet me. My name is John.” 
2. The agent sees that it can call `greet(name=str)`.
3. If the agent decides to call it, tangent will pass `"John"` into the function `greet("John")`.
4. The function returns `"Hello, John! This greeting is brought to you by the greet() function."`
5. That result is appended to the conversation as a tool message, and the agent may finalize a response to the user.

---

## 8. Notes & Best Practices

- **Model**: By default, Agents use `model="gpt-4o"`. You can override this via the `model` field in the `Agent` constructor, for instance `model="gpt-4o-0125-preview"`.
- **Parallel Tool Calls**: The default for an `Agent` is `parallel_tool_calls=True`. If you only want your agent to call one function at a time, set `parallel_tool_calls=False`.
- **Error Handling**: If an Agent calls a function that doesn’t exist or passes bad arguments, the library appends an error to the conversation so the model can gracefully recover.
- **Handoffs**: The last function call that returns an `Agent` will finalize the handoff. If multiple tools are called in sequence, the final one returning an `Agent` is the one that decides the new active agent.

---

### Summary

**Creating an Agent** in **tangent** boils down to:
1. Importing and instantiating `Agent` from `tangent`.
2. (Optionally) providing `instructions` (string or function).
3. (Optionally) attaching **function(s)** in `Agent.functions`.
4. Optionally using `tangent().run()` with user messages to see the agent in action.

This covers the entire process of building a single- or multi-function agent using the code found in the tangent library. 
```

## docs/embeddings.md

```markdown
Below is a **complete and self-contained explanation** of how the tangent Python library **uses the `text-embedding-3-large` (and similarly `text-embedding-3-small`) embeddings models**, based **entirely** on the provided documentation. This guide will detail **all the ways** these embeddings can be utilized in a tangent-based project, along with **every step** present in the example codebases that show how these embeddings are created, stored, and retrieved.

---

## 1. Where `text-embedding-3-large` Appears in tangent

In the **examples** directory, you can see references to the **embedding model** parameter set to `"text-embedding-3-large"`. Specifically:

- **`examples/customer_service_streaming/prep_data.py`**  
- **`examples/customer_service_streaming/configs/tools/query_docs/handler.py`**  
- **`examples/support_bot/prep_data.py`**  

In each of these, **OpenAI’s** embeddings endpoint is used to transform text into high-dimensional vectors. The code calls:

```python
client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=x['text']
)
```

Here, `EMBEDDING_MODEL` is the string `'text-embedding-3-large'` or `'text-embedding-3-small'`.

---

## 2. Overview: What Happens With Embeddings in tangent Projects

1. **Data is Preprocessed**: In the examples, JSON articles or documents are loaded from disk, and each article’s text is run through the embedding model (`text-embedding-3-large` or `text-embedding-3-small`) to create vector embeddings.  
2. **Vectors are Stored in Qdrant**: tangent can integrate with **Qdrant**—a vector database. The code indexes these embeddings along with their metadata (title, text, etc.) into a Qdrant collection.  
3. **Querying by Embeddings**: When an end user poses a question, tangent calls `query_qdrant`, which again uses the embedding model to transform the user’s query text into a vector, then performs a nearest-neighbor search in Qdrant. The best matching articles are returned.  
4. **Use in Agents**: Various tangent agents or tools (e.g., `query_docs`) can rely on this embedding-based search to help answer user questions about the stored data.

---

## 3. Creating Embeddings During Data “Prep” Stage

### Example: `examples/customer_service_streaming/prep_data.py`

```python
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-large"

article_list = os.listdir('data')
articles = []

# Load each article JSON
for x in article_list:
    article_path = 'data/' + x
    f = open(article_path)
    data = json.load(f)
    articles.append(data)
    f.close()

# Generate embeddings
for i, x in enumerate(articles):
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=x['text'])
        articles[i].update({"embedding": embedding.data[0].embedding})
    except Exception as e:
        print(x['title'])
        print(e)
```

**Step-by-Step**:

1. The code loops over JSON files in a `data/` directory, each containing article text.  
2. Each article’s text is passed to `client.embeddings.create(...)`, specifying `"text-embedding-3-large"` as the model.  
3. The resulting embedding (a list of floating-point numbers) is added to the article dictionary under the key `"embedding"`.

### 3.1 Inserting Embeddings Into Qdrant

Immediately after generating each embedding, the example code stores them in a Qdrant vector database:

```python
qdrant.upsert(
    collection_name=collection_name,
    points=[
        rest.PointStruct(
            id=k,
            vector={'article': v['embedding']},
            payload=v.to_dict(),
        )
        for k, v in article_df.iterrows()
    ],
)
```

- A new Qdrant collection is created or re-created to hold these vectors.  
- Each article’s embedding becomes the `'article'` vector.  
- All the textual metadata (title, text, etc.) goes into `payload`.

By doing this, an entire corpus of documents is now searchable via vector embeddings.

---

## 4. Using Embeddings for Document Retrieval

### Example: `examples/customer_service_streaming/configs/tools/query_docs/handler.py`

```python
EMBEDDING_MODEL = 'text-embedding-3-large'

def query_docs(query):
    print(f'Searching knowledge base with query: {query}')
    query_results = query_qdrant(query, collection_name=collection_name)
    ...
```

And inside **`query_qdrant`**:

```python
def query_qdrant(query, collection_name, vector_name='article', top_k=5):
    # Convert user query to embedding
    embedded_query = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    ).data[0].embedding

    # Perform vector search in Qdrant
    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
    )
    return query_results
```

Here’s the **flow**:

1. **User** interacts with a tangent agent that calls `query_docs(...)`.  
2. The function **re-embeds** the user’s query into a vector using `"text-embedding-3-large"`.  
3. That vector is used to **search** the Qdrant index, retrieving the top matching articles.  
4. The matching article(s) are returned to the agent, which can integrate that content into a response.

---

## 5. Additional Notes on Usage With tangent

- **Agent Tools**: In `assistant.json` or in your `Agent` definition, you may define a function like `query_docs`. That function is recognized as a “tool” by the tangent framework.  
- **Parallel or Single Tool Calls**: Agents in tangent can either call multiple tools in parallel or call them sequentially (depending on the `parallel_tool_calls` setting). This includes the embedding-based tool if the conversation demands it.  
- **Combining With Other Tools**: Once data is retrieved via embeddings, you might chain it with a second tool, such as `send_email`, to mail the user’s requested information.  
- **Scaling**: Whether you use `"text-embedding-3-large"` or `"text-embedding-3-small"`, the code pattern remains the same. You set `EMBEDDING_MODEL` in your Python script, then call `client.embeddings.create(model=EMBEDDING_MODEL, input=...)`.

---

## 6. Summary of Steps

1. **Set EMBEDDING_MODEL** (e.g., `"text-embedding-3-large"`) in your script.  
2. **Load your documents** from JSON files or other data sources.  
3. **Generate embeddings** for each document by calling:  
   ```python
   embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=document_text)
   ```
4. **Store** those embeddings into a vector database (like Qdrant) along with the documents.  
5. **At query time**, embed the user’s query text with the **same** model, and pass that vector to Qdrant for nearest-neighbor search.  
6. **Return** the most relevant docs to the user or agent.  

Hence, tangent’s usage of `text-embedding-3-large` (or `text-embedding-3-small`) centers on:

- **Preprocessing** text into vectors  
- **Storing** those vectors in Qdrant  
- **Searching** that vector space to find relevant information for the user  

Everything is orchestrated seamlessly within the multi-agent environment that tangent provides.

---

**That’s all, based entirely on the provided tangent documentation.**
```

## docs/function_calls.md

```markdown
Below is a step-by-step explanation of **how the tangent library handles function calls** and **how to create them** within your code, drawn **exclusively** from the tangent source code provided. This guide aims to be complete, clear, and easy to follow, **showing you every step** involved in registering and calling functions from within tangent.

---

## 1. Overview of Function Calls in tangent

In tangent, **functions** (sometimes called “tools”) are Python callables that an agent can invoke in response to user messages. When the model decides it needs to perform an action—like “get the weather,” “query a database,” or “send an email”—it emits a **function call**. tangent then **maps** that function call to your actual Python function, executes it, and **inserts** the result back into the conversation.

---

## 2. The `Agent` Object and Its Functions

The core concept in tangent is an **`Agent`**. An `Agent` can have a list of Python functions that it is allowed to call. These are stored in the agent’s `functions` attribute. For example:

```python
from tangent import Agent

def get_weather(location):
    """
    Return the weather for a given location.
    """
    return f"The weather in {location} is sunny."

weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent that can report the weather.",
    functions=[get_weather],
)
```

Here:

- We define `get_weather(...)` as a Python function that returns simple weather info.  
- We **attach** that function to `weather_agent` via `functions=[get_weather]`.

---

## 3. How tangent Describes Your Functions Internally

When you pass a Python function to an `Agent`, tangent automatically:

1. **Inspects** its signature (the function name, the parameter names, etc.)  
2. **Generates** a JSON schema describing the function, using the logic in **`tangent/util.py -> function_to_json`**.  

In `function_to_json`, tangent uses Python’s `inspect.signature(func)` to gather:

- **The function’s name**  
- **Parameter names**  
- **Parameter types** (for example, `str`, `int`)  

and turns that into a JSON structure that can be passed to the model. The relevant snippet in **`tangent/util.py`** is:

```python
def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.
    """
    # ...
    signature = inspect.signature(func)
    parameters = {}
    # For each parameter in your function signature, gather type info:
    # Build JSON schema from that type info.
    # ...
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

This JSON schema is how the model “knows” how to call your function.  

---

## 4. Passing Functions to the Model

Within **`tangent/core.py`**, look at the method **`get_chat_completion`**. tangent collects each Python function you attached to the `Agent`, converts it to JSON with `function_to_json`, and sends that along in the `tools` parameter:

```python
tools = [function_to_json(f) for f in agent.functions]
create_params = {
    "model": model_override or agent.model,
    "messages": messages,
    "tools": tools or None,
    "tool_choice": agent.tool_choice,
    "stream": stream,
}
# ...
return self.client.chat.completions.create(**create_params)
```

The model sees your function definitions (in JSON), and if it wants to call a function, it responds with a **tool call** describing which function it’s calling and what arguments it’s passing.

---

## 5. Handling the Model’s Function Calls

When the model emits a function call, tangent processes it in **`tangent/core.py -> handle_tool_calls`**. The steps are:

1. The response from OpenAI might say something like “Call function: `get_weather(location='New York')`.”  
2. tangent loops through each requested function name, looks it up in the dictionary of your provided Python functions, and then loads the JSON arguments.  
3. It **executes** that Python function with the JSON arguments.  
4. The return value is wrapped in a `Result` object or a plain string.  

This section in `handle_tool_calls` is key:

```python
def handle_tool_calls(
    self,
    tool_calls: List[ChatCompletionMessageToolCall],
    functions: List[AgentFunction],
    context_variables: dict,
    debug: bool,
) -> Response:
    function_map = {f.__name__: f for f in functions}
    # ...
    for tool_call in tool_calls:
        name = tool_call.function.name
        # find the actual Python function by name
        func = function_map[name]
        args = json.loads(tool_call.function.arguments)
        # ...
        raw_result = function_map[name](**args)  # Actual Python call
        result = self.handle_function_result(raw_result, debug)
        # ...
    # ...
```

So effectively:

1. **Model** says: “I want to call `get_weather` with location='NYC'.”  
2. **tangent** runs `get_weather(location='NYC')`.  
3. The result is appended as a new “tool” message in the conversation.

---

## 6. Creating a Function for tangent Step by Step

Below is an **explicit** step-by-step guide to set up your own function calls:

1. **Write your Python function**.  
   - It can accept zero or more parameters.  
   - Return a string, a dictionary, or a `Result` object.

2. **Give it a docstring**.  
   - The docstring becomes the “description” that the model sees.  
   - Example:
     ```python
     def greet(name: str):
         """
         Greet a user by name.
         """
         return f"Hello, {name}!"
     ```

3. **Attach it to your `Agent`**.  
   ```python
   from tangent import Agent

   my_agent = Agent(
       name="Greeting Agent",
       instructions="You greet users warmly.",
       functions=[greet],
   )
   ```

4. **Use the `tangent` client to run**.  
   ```python
   from tangent import tangent

   client = tangent()

   messages = [{"role": "user", "content": "Hi, I'm Alice."}]
   response = client.run(agent=my_agent, messages=messages)
   ```

5. **During the conversation**, if the model calls `greet(name='Alice')`, tangent:

   - Finds `greet` in the agent’s function list.  
   - Invokes `greet(name='Alice')`.  
   - Inserts that result as a tool message.  

---

## 7. Observing the Flow in Practice

Let’s do a condensed example:

```python
from tangent import tangent, Agent

def greet_user(name: str) -> str:
    """
    Greet a user by name with a friendly message.
    """
    return f"Hello, {name}! How can I help you today?"

agent = Agent(
    name="Greeter",
    instructions="You are a helpful assistant that greets users by name.",
    functions=[greet_user],
)

client = tangent()

messages = [{"role": "user", "content": "My name is Bob. Greet me."}]
response = client.run(agent=agent, messages=messages)

print(response.messages)  # Inspect the conversation and see the tool call and response.
```

- The OpenAI model sees the `greet_user` function in JSON.  
- If it chooses to call it, you’ll see a tool call like:

  ```json
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "greet_user",
          "arguments": "{\"name\":\"Bob\"}"
        },
        ...
      }
    ]
  }
  ```

- tangent’s `handle_tool_calls` runs `greet_user(name='Bob')`.  
- The function’s return shows up as a `tool` role message in the conversation.  

---

## 8. Returning a Custom `Result`

Sometimes, you want your function to return extra data, like updated context variables. You can do that using the `Result` model from **`tangent/types.py`**:

```python
from tangent.types import Result

def greet_user(name: str):
    """
    Greet a user by name. Also store the name in context.
    """
    return Result(
        value=f"Hello, {name}!",
        context_variables={"user_name": name}
    )
```

When tangent executes that function, the `context_variables` get merged back into the conversation’s context.  

---

## 9. Debugging Your Function Calls

If you enable `debug=True` in `client.run(...)`, you’ll see logs about:

- Which function is called  
- The arguments that were passed  
- The raw results  

For instance:
```python
response = client.run(
    agent=agent,
    messages=messages,
    debug=True,
)
```
The console will show lines like:
```
[DEBUG] Processing tool call: greet_user with arguments {'name': 'Bob'}
```

---

## 10. Summary

- **Register** your Python function(s) by adding them to an `Agent`’s `functions` list.  
- tangent **converts** each function into a JSON schema.  
- When the model **calls** the function, tangent automatically **executes** it and returns the result to the conversation.  
- You can return a plain string or a `Result` object to store updated context variables, an updated agent, or more.  

This is the complete, step-by-step approach to **how tangent manages function calls** and **how you can create them** in your own code. By following these guidelines, you’ll be able to leverage tangent’s multi-agent orchestration framework and give each agent flexible tools to perform tasks and respond to user queries.
```

## my-docs/tangent_python_library.md

```markdown
Here's a quick summary of your core files in the tangent library:

1. **`types.py`**
- Defines the core data models using Pydantic
- Contains three main classes:
  - `Agent`: Configuration for AI agents (name, model, instructions, available functions)
  - `Response`: Holds conversation results (messages, current agent, context variables)
  - `Result`: Encapsulates function return values (value, optional agent, context variables)

2. **`util.py`**
- Provides utility functions for the library
- Key functions:
  - `debug_print`: Handles formatted debug output with timestamps
  - `merge_fields`/`merge_chunk`: Manages streaming response merging
  - `function_to_json`: Converts Python functions to OpenAI-compatible JSON schemas

3. **`core.py`**
- The heart of the library - implements the `tangent` class
- Handles:
  - Chat completions
  - Tool/function
  - Conversation state management
  - streaming/non-streaming modes
  - Agent switching/context variable management

4. **`repl/repl.py`**
- CLI for interacting with the library
- Streaming response processing
- Interactive loop for agent conversations
- Clean output formatting for tool calls and responses
```

## examples/websearch_agent.py

```python
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
```

