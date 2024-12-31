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