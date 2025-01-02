# Quick Start

## Create a new agent
```python
from tangent import tangent, Agent, run_tangent_loop

client = tangent()

agent = Agent(
    name="Basic Agent",
    model="gpt-4o",
    instructions="You are a simple chatbot that can respond to user requests."
)

run_tangent_loop(agent, stream=True, debug=False)
```

# tangent Python package - Detailed Overview

## package file structure
```
 tangent/
[A] ├── types.py 
[B] ├── core.py
[C] ├── util.py
[D] ├── embeddings.py
[E] ├── __init__.py 
    ├── repl/
[F] │   ├── __init__.py
[G] │   └── repl.py
    ├── tools/
[H] │   ├── __init__.py
[I] │   └── knowledge_base.py
    ├── triage/
[J] │   ├── __init__.py
[K] │   ├── agent.py
[L] │   └── utils.py
```

## Imports *(all are 'from tangent import ...')*

### [E] Core Components - *created in [B] core.py*:
   - `tangent`: The main class for handling AI agent interactions
      ```python
      from tangent import tangent
      ```
   - `Agent`: Base class for creating AI agents
      ```python
      from tangent import Agent
      ```
   - `Response`: Class for encapsulating agent responses
      ```python
      from tangent import Response
      ```
   - `Result`: Class for encapsulating function return values
      ```python
      from tangent import Result
      ```

### [E] Embedding Components - *created in [D] embeddings.py*:
   - `DocumentStore`: High-level interface for document management and search
      ```python
      from tangent import DocumentStore
      ```

### [F] REPL Components - *created in [G] repl/repl.py*:
   - `run_tangent_loop`: Function to run an interactive agent session
      ```python
      from tangent import run_tangent_loop
      ```
   - `process_and_print_streaming_response`: Function to handle streaming responses
      ```python
      from tangent import process_and_print_streaming_response
      ```

### [H] Tools Components - *created in [I] tools/knowledge_base.py*:
   - `search_knowledge_base`: Function to search through an agent's knowledge base
      ```python
      from tangent import search_knowledge_base
      ```

### [J] Triage Components - *created in [K] triage/agent.py and [L] triage/utils.py*:
   - `create_triage_agent`: Function to create a triage agent
      ```python
      from tangent import create_triage_agent
      ```

## [A] tangent.types

### Core Agent Types - *created in [B] core.py*

1. `Agent`
   - **Type**: Pydantic BaseModel
   - **Purpose**: Represents an AI agent with specific capabilities
   - **Fields**:
     - `name: str` - Agent name (default: "Agent")
     - `model: str` - Model identifier (default: "gpt-4o")
     - `instructions: Union[str, Callable[[], str]]` - Agent instructions
     - `functions: List[AgentFunction]` - Available functions (default: [])
     - `tool_choice: str` - Tool selection preference (default: None)
     - `parallel_tool_calls: bool` - Enable parallel tool calls (default: True)
     - `triage_assignment: Optional[str]` - Triage agent assignment (optional)
     - `embedding_config: Optional[str]` - Embedding configuration (optional)
     - `embedding_manager: Optional[str]` - Embedding manager (optional)

2. `Response`
   - **Type**: Pydantic BaseModel
   - **Purpose**: Encapsulates agent response data
   - **Fields**:
     - `messages: List` - Response messages (default: [])
     - `agent: Optional[Agent]` - Associated agent (optional)
     - `context_variables: dict` - Context variables (default: {})

3. `Result`

    - **Type**: Pydantic BaseModel
    - **Purpose**: Encapsulates function return values
    - **Fields**:
      - `value: str` - Result value (default: "")
      - `agent: Optional[Agent]` - Associated agent (optional)
      - `context_variables: dict` - Context variables (default: {})

### Document Types - *created in [D] embeddings.py*

1. `DocumentChunk`
   - **Type**: dataclass
   - **Purpose**: Represents a chunk of text from a document with its metadata
   - **Fields**:
     - `text: str` - The actual text content
     - `metadata: dict` - Associated metadata
     - `source_file: str` - Source file path
     - `chunk_index: int` - Index of the chunk (default: 0)

2. `Document`
   - **Type**: Pydantic BaseModel
   - **Purpose**: Represents a complete document with text and metadata
   - **Fields**:
     - `id: str` - Document identifier
     - `text: str` - Document content
     - `metadata: dict` - Associated metadata (default: {})
     - `embedding: Optional[List[float]]` - Vector embedding (optional)

### Embedding Configuration - *created in [D] embeddings.py*

7. `EmbeddingConfig`
   - **Type**: Pydantic BaseModel
   - **Purpose**: Configuration for embedding functionality
   - **Fields**:
     - `model: str` - Embedding model name (default: "text-embedding-3-large")
     - `chunk_size: int` - Size of text chunks (default: 500)
     - `chunk_overlap: int` - Overlap between chunks (default: 50)
     - `batch_size: int` - Batch size for processing (default: 100)
     - `vector_db: Union[QdrantConfig, PineconeConfig, CustomVectorDBConfig]` - Vector database configuration
     - `supported_extensions: List[str]` - Supported file extensions
     - `recreate_collection: bool` - Whether to recreate collection (default: False)

### Vector Database Configuration Types - *created in [D] embeddings.py*

3. `VectorDBConfig`
   - **Type**: Pydantic BaseModel
   - **Purpose**: Base configuration for vector databases
   - **Fields**:
     - `type: Literal["qdrant", "pinecone", "custom"]` - Database type (default: "qdrant")
     - `collection_name: str` - Name of collection (default: "default")

4. `QdrantConfig`
   - **Type**: VectorDBConfig
   - **Purpose**: Qdrant-specific configuration
   - **Fields**:
     - `type: Literal["qdrant"]` - Fixed as "qdrant"
     - `url: str` - Server URL (default: "localhost")
     - `port: int` - Server port (default: 6333)
     - `api_key: Optional[str]` - API key if required

5. `PineconeConfig`
   - **Type**: VectorDBConfig
   - **Purpose**: Pinecone-specific configuration
   - **Fields**:
     - `type: Literal["pinecone"]` - Fixed as "pinecone"
     - `api_key: str` - Pinecone API key
     - `environment: str` - Pinecone environment
     - `index_name: str` - Name of the Pinecone index

6. `CustomVectorDBConfig`
   - **Type**: VectorDBConfig
   - **Purpose**: Configuration for custom vector database implementations
   - **Fields**:
     - `type: Literal["custom"]` - Fixed as "custom"
     - `connection_params: dict` - Custom connection parameters (default: {})

### Type Aliases

- `AgentFunction = Callable[..., Union[str, 'Agent', dict, 'Result']]` - Type alias for agent functions
- `EmbeddingManager = ForwardRef('EmbeddingManager')` - Forward reference for EmbeddingManager

## Triage Components

### How triage agent works:
1. Create a triage agent using `create_triage_agent`
2. System discovers or is given managed agents
3. Transfer functions are created for each managed agent
4. When running:
   - Triage agent analyzes requests
   - Can transfer to specialized agents
   - Specialized agents can transfer back

### [K] Triage Agent
This is the core triage functionality with several key functions:

- `create_transfer_functions(managed_agents, triage_agent)`:
  - Creates functions that allow transferring between agents
  - Makes a transfer function for each managed agent
  - Adds "transfer back" functions to managed agents

- `discover_assigned_agents(triage_agent_name)`:
  - Finds all agents assigned to a specific triage agent
  - Searches through all loaded Python modules
  - Looks for agents with matching `triage_assignment`

- `enhance_instructions(base_instructions, managed_agents)`:
  - Enhances the triage agent's instructions
  - Adds information about available specialized agents
  - Includes guidance on when to transfer

- `create_triage_agent(...)`:
  - Main function to create a triage agent
  - Can auto-discover assigned agents
  - Sets up transfer functions
  - Configures enhanced instructions

### [L] Triage Utils
This is a utility file with three main functions for managing triage agents:

- `discover_agents(module)`:
  - Scans a Python module to find all Agent instances
  - Returns a dictionary of agent names to Agent objects
  - Used for automatic agent discovery

- `validate_agent_compatibility(agent)`:
  - Checks if an agent can work with the triage system
  - Verifies the agent has required attributes (name, instructions, functions)
  - Returns True/False for compatibility

- `generate_agent_description(agent)`:
  - Creates a human-readable description of an agent
  - Includes the agent's name, instructions, and capabilities
  - Extracts docstrings from the agent's functions for capability descriptions