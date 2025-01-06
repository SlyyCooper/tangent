from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional, Dict, Literal, ForwardRef, TYPE_CHECKING, Any
from pydantic import BaseModel, Field, ConfigDict
import uuid
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .embeddings import EmbeddingManager

class Structured_Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        extracted_data (dict): A dictionary of context variables.
    """
    value: str = ""
    agent: Optional['Agent'] = None
    extracted_data: dict = {}

# Type alias for agent functions
AgentFunction = Callable[..., Union[str, 'Agent', dict, Structured_Result]]

# 1. Document Types
@dataclass
class DocumentChunk:
    """A chunk of text from a document with its metadata."""
    text: str
    metadata: dict
    source_file: str
    chunk_index: int = 0

class Document(BaseModel):
    """Represents a document with its text and metadata."""
    id: str
    text: str
    metadata: dict = {}
    embedding: Optional[List[float]] = None

# 2. Vector Database Configuration Types
class VectorDBConfig(BaseModel):
    """Base configuration for vector databases."""
    type: Literal["qdrant", "pinecone", "custom"] = "qdrant"
    collection_name: str = "default"

class QdrantConfig(VectorDBConfig):
    """Qdrant-specific configuration."""
    type: Literal["qdrant"] = "qdrant"
    url: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None

class PineconeConfig(VectorDBConfig):
    """Pinecone-specific configuration."""
    type: Literal["pinecone"] = "pinecone"
    api_key: str
    environment: str
    index_name: str

class CustomVectorDBConfig(VectorDBConfig):
    """Configuration for custom vector database implementations."""
    type: Literal["custom"] = "custom"
    connection_params: dict = {}

# 3. Embedding Configuration
class EmbeddingConfig(BaseModel):
    """Configuration for embedding functionality."""
    model: str = "text-embedding-3-large"
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 100
    vector_db: Union[QdrantConfig, PineconeConfig, CustomVectorDBConfig] = Field(
        default_factory=lambda: QdrantConfig(collection_name="default")
    )
    supported_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".json"]
    recreate_collection: bool = False

# 4. Core Agent Types
class InstructionsSource(str, Enum):
    """Source type for agent instructions."""
    INLINE = "inline"      # Direct string in code
    CALLABLE = "callable"  # Function that returns string
    FILE = "file"         # Load from instructions/ directory

class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str], Literal["from_file"]] = "You are a helpful agent."
    instructions_source: InstructionsSource = InstructionsSource.INLINE
    instructions_dir: str = "instructions"  # Directory to load instructions from
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    triage_assignment: Optional[str] = Field(None, description="Name of the triage agent this agent is assigned to")
    embedding_config: Optional[str] = None
    embedding_manager: Optional[Any] = None  # Use Any for runtime type checking
    
    def setup_embeddings(self, config: 'EmbeddingConfig') -> 'Agent':
        """Initialize embedding support for this agent."""
        self.embedding_config = config
        from .embeddings import EmbeddingManager  # Import here to avoid circular imports
        self.embedding_manager = EmbeddingManager(config)
        return self
    
    def load_knowledge_base(self, source: str, path: str) -> 'Agent':
        """Load and embed documents for the agent's knowledge base."""
        if not self.embedding_manager:
            raise ValueError("Embeddings not configured. Call setup_embeddings first.")
            
        # Load documents
        documents = self.embedding_manager.load_documents(source, path)
        
        # Create embeddings
        documents = self.embedding_manager.create_embeddings(documents)
        
        # Store in vector database
        self.embedding_manager.store_documents(documents)
        
        return self
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List['Document']:
        """Search the agent's knowledge base."""
        if not self.embedding_manager:
            raise ValueError("Embeddings not configured. Call setup_embeddings first.")
            
        return self.embedding_manager.search(query, top_k)

class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    extracted_data: dict = {}