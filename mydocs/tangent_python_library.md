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