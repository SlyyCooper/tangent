# Current Tangent Importable Components (`from tangent import ...`)

## Currently Available Imports in `tangent`

### Core Components
1. `tangent`
   - The main client class
   - Used to create the base client for interacting with AI models

2. `Agent`
   - Base class for creating AI agents
   - Handles model settings, instructions, and functions

3. `Response`
   - Type for encapsulating agent responses
   - Contains messages, agent state, and context variables

4. `Result`
   - Type for function return values
   - Contains value, agent, and context variables

### REPL/CLI Components
5. `run_chat_loop`
   - Main CLI interface
   - Handles input/output in terminal

6. `process_and_print_streaming_response`
   - Handles streaming responses
   - Formats and prints responses in real-time

### Special Features
7. `create_triage_agent`
   - Creates agents that can route requests
   - Manages multiple specialized agents

8. `DocumentStore`
   - Handles document management
   - For knowledge base features

9. `search_knowledge_base`
   - Searches through embedded documents
   - Part of the knowledge base system