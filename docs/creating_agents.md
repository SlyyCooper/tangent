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
- **instructions** become the `system` prompt. They can be a **string** or a **callable** (i.e., a function) that returns a string dynamically. If you pass in a callable, it may optionally take `extracted_data` as a parameter to inject dynamic context.

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

If you want your agent’s instructions to be dynamic, you can define `instructions` as a function that optionally accepts `extracted_data`. For example:

```python
def dynamic_instructions(extracted_data):
    user_name = extracted_data.get("user_name", "user")
    return f"You are a helpful agent. Always greet {user_name} politely."

my_dynamic_agent = Agent(
    name="Dynamic Agent",
    instructions=dynamic_instructions
)
```

Then, when you run the agent, you supply `extracted_data`:

```python
response = client.run(
    agent=my_dynamic_agent,
    messages=[{"role": "user", "content": "Hi!"}],
    extracted_data={"user_name": "Alice"}
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