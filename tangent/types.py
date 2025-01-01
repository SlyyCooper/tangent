from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional
from pydantic import BaseModel

from .embeddings import EmbeddingConfig, EmbeddingManager, Document

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    
    # New embedding fields
    embedding_config: Optional[EmbeddingConfig] = None
    _embedding_manager: Optional[EmbeddingManager] = None
    
    def setup_embeddings(self, config: EmbeddingConfig):
        """Initialize embedding support for this agent."""
        self.embedding_config = config
        self._embedding_manager = EmbeddingManager(config)
        return self
    
    def load_knowledge_base(self, source: str, path: str):
        """Load and embed documents for the agent's knowledge base."""
        if not self._embedding_manager:
            raise ValueError("Embeddings not configured. Call setup_embeddings first.")
            
        # Load documents
        documents = self._embedding_manager.load_documents(source, path)
        
        # Create embeddings
        documents = self._embedding_manager.create_embeddings(documents)
        
        # Store in vector database
        self._embedding_manager.store_documents(documents)
        
        return self
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Document]:
        """Search the agent's knowledge base."""
        if not self._embedding_manager:
            raise ValueError("Embeddings not configured. Call setup_embeddings first.")
            
        return self._embedding_manager.search(query, top_k)


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