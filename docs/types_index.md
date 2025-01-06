From `types.py`, here are all the current types/classes:

1. **Document Types**
```python
@dataclass
class DocumentChunk:
    text: str
    metadata: dict
    source_file: str
    chunk_index: int = 0

class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}
    embedding: Optional[List[float]] = None
```

2. **Vector Database Configuration Types**
```python
class VectorDBConfig(BaseModel):
    type: Literal["qdrant", "pinecone", "custom"] = "qdrant"
    collection_name: str = "default"

class QdrantConfig(VectorDBConfig):
    type: Literal["qdrant"] = "qdrant"
    url: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None

class PineconeConfig(VectorDBConfig):
    type: Literal["pinecone"] = "pinecone"
    api_key: str
    environment: str
    index_name: str

class CustomVectorDBConfig(VectorDBConfig):
    type: Literal["custom"] = "custom"
    connection_params: dict = {}
```

3. **Embedding Configuration**
```python
class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-large"
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 100
    vector_db: Union[QdrantConfig, PineconeConfig, CustomVectorDBConfig] = Field(
        default_factory=lambda: QdrantConfig(collection_name="default")
    )
    supported_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".json"]
    recreate_collection: bool = False
```

4. **Type Alias: AgentFunction**
```python
AgentFunction = Callable[[], Union[str, "Agent", dict]]
```

5. **Class: Agent**
```python
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]]
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    embedding_config: Optional[EmbeddingConfig] = None
    _embedding_manager: Optional[EmbeddingManager] = None
```

6. **Class: Response**
```python
class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    extracted_data: dict = {}
```

7. **Class: Result**
```python
class Result(BaseModel):
    value: str = ""
    agent: Optional[Agent] = None
    extracted_data: dict = {}
```

The file also imports these types from OpenAI:
- `ChatCompletionMessage`
- `ChatCompletionMessageToolCall`
- `Function`

And from embeddings:
- `EmbeddingConfig`
- `EmbeddingManager`
- `Document`

