from typing import List, Optional, Union, Dict
from pathlib import Path
import json
import os
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from pydantic import BaseModel

# Known vector sizes for embedding models
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

class Document(BaseModel):
    """Represents a document with its text and metadata."""
    id: str
    text: str
    metadata: dict = {}
    embedding: Optional[List[float]] = None

class EmbeddingConfig(BaseModel):
    """Configuration for embedding functionality."""
    model: str = "text-embedding-3-large"
    collection_name: str = "default"
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    chunk_size: int = 500
    batch_size: int = 100
    recreate_collection: bool = False  # Whether to recreate collection if it exists

class EmbeddingManager:
    """Manages document loading, embedding creation, and search."""
    
    def __init__(self, config: EmbeddingConfig, client=None):
        self.config = config
        self.client = client or OpenAI()
        self.qdrant = QdrantClient(
            host=config.qdrant_url,
            port=config.qdrant_port
        )
        
        # Get vector size for the model
        if config.model not in EMBEDDING_DIMENSIONS:
            # If model not known, determine size dynamically
            test_response = self.client.embeddings.create(
                model=self.config.model,
                input="test"
            )
            self.vector_size = len(test_response.data[0].embedding)
        else:
            self.vector_size = EMBEDDING_DIMENSIONS[config.model]
        
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup the vector collection, creating or recreating as needed."""
        try:
            collection_info = self.qdrant.get_collection(self.config.collection_name)
            collection_exists = True
            
            # Check if existing collection has correct vector size
            if collection_info.config.params.vectors.size != self.vector_size:
                if not self.config.recreate_collection:
                    raise ValueError(
                        f"Existing collection '{self.config.collection_name}' has incorrect vector size "
                        f"({collection_info.config.params.vectors.size} vs {self.vector_size}). "
                        f"Set recreate_collection=True in config to recreate the collection."
                    )
                self.qdrant.delete_collection(self.config.collection_name)
                collection_exists = False
                
        except Exception as e:
            if "Not found" not in str(e):
                raise
            collection_exists = False
        
        if not collection_exists or self.config.recreate_collection:
            try:
                self.qdrant.delete_collection(self.config.collection_name)
            except:
                pass
            self.qdrant.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance.COSINE
                )
            )

    def load_documents(self, source: str, path: str) -> List[Document]:
        """Load documents from various sources."""
        documents = []
        
        if source == "folder":
            path = Path(path)
            for file in path.glob("**/*.json"):
                with open(file) as f:
                    data = json.load(f)
                    doc = Document(
                        id=str(file),
                        text=data.get("text", ""),
                        metadata=data
                    )
                    documents.append(doc)
                    
        return documents
    
    def create_embeddings(self, documents: List[Document]) -> List[Document]:
        """Create embeddings for documents in batches."""
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            texts = [doc.text for doc in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
                
                for doc, embedding_data in zip(batch, response.data):
                    doc.embedding = embedding_data.embedding
                    print(f"Document embedding size: {len(doc.embedding)}")
                    
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                
        return documents
    
    def store_documents(self, documents: List[Document]):
        """Store documents and their embeddings in Qdrant."""
        points = [
            rest.PointStruct(
                id=str(uuid.uuid4()),  # Use UUID instead of hash
                vector=doc.embedding,
                payload={
                    "text": doc.text,
                    "metadata": doc.metadata
                }
            )
            for doc in documents
            if doc.embedding
        ]
        
        self.qdrant.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for similar documents."""
        # Create query embedding
        query_response = self.client.embeddings.create(
            model=self.config.model,
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Convert results back to Documents
        documents = []
        for result in results:
            doc = Document(
                id=str(result.id),
                text=result.payload["text"],
                metadata=result.payload["metadata"],
                embedding=result.vector
            )
            documents.append(doc)
            
        return documents 