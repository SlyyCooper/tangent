from tangent import Agent, tangent
from tangent.types import Result
from tangent.embeddings import Document, EmbeddingConfig, EmbeddingManager
from typing import List, Dict, Optional
import os
from pathlib import Path
import json
import uuid
from qdrant_client.http import models as rest

class DocumentEmbeddings:
    def __init__(self):
        self.embedding_manager = EmbeddingManager(
            EmbeddingConfig(
                model="text-embedding-3-large",
                collection_name="ai_documents",
                recreate_collection=True  # Start fresh for the example
            )
        )
        self.docs_dir = Path("document_storage")
        self.docs_dir.mkdir(exist_ok=True)
        
        # Ensure collection exists
        try:
            self.embedding_manager.qdrant.get_collection(self.embedding_manager.config.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.embedding_manager.qdrant.create_collection(
                collection_name=self.embedding_manager.config.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.embedding_manager.vector_size,
                    distance=rest.Distance.COSINE
                )
            )

    def embed_text(self, title: str, content: str, metadata: dict = None) -> Document:
        """Embed a single document with title and content."""
        doc = Document(
            id=str(uuid.uuid4()),
            text=content,
            metadata={"title": title, **(metadata or {})}
        )
        
        # Save document to disk
        doc_path = self.docs_dir / f"{doc.id}.json"
        with open(doc_path, "w") as f:
            json.dump({
                "id": doc.id,
                "title": title,
                "content": content,
                "metadata": metadata or {}
            }, f)
        
        # Create embedding and store
        [doc_with_embedding] = self.embedding_manager.create_embeddings([doc])
        self.embedding_manager.store_documents([doc_with_embedding])
        
        return doc_with_embedding

    def embed_file(self, file_path: str) -> List[Document]:
        """Embed documents from a file or directory."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        if path.is_file():
            if path.suffix != '.json':
                raise ValueError("Only JSON files are supported")
            with open(path) as f:
                data = json.load(f)
                return [self.embed_text(
                    data.get("title", path.stem),
                    data.get("content", data.get("text", "")),
                    data.get("metadata", {})
                )]
        else:
            documents = []
            for json_file in path.glob("**/*.json"):
                with open(json_file) as f:
                    data = json.load(f)
                    doc = self.embed_text(
                        data.get("title", json_file.stem),
                        data.get("content", data.get("text", "")),
                        data.get("metadata", {})
                    )
                    documents.append(doc)
            return documents

    def semantic_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Search documents using semantic similarity."""
        return self.embedding_manager.search(query, top_k=top_k)

# Initialize document embeddings
doc_embedder = DocumentEmbeddings()

def embed_document(title: str, content: str, metadata: str = None) -> Result:
    """
    Embed a single document with title and content.
    """
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        doc = doc_embedder.embed_text(title, content, metadata_dict)
        return Result(
            value=f"Successfully embedded document '{title}' with ID {doc.id}",
            context_variables={
                "last_document": {
                    "id": doc.id,
                    "title": title,
                    "embedding_size": len(doc.embedding)
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error embedding document: {str(e)}")

def embed_from_file(file_path: str) -> Result:
    """
    Embed documents from a JSON file or directory of JSON files.
    Each file should have 'title', 'content', and optional 'metadata'.
    """
    try:
        docs = doc_embedder.embed_file(file_path)
        return Result(
            value=f"Successfully embedded {len(docs)} documents from {file_path}",
            context_variables={
                "embedded_docs": [
                    {"id": doc.id, "title": doc.metadata.get("title")}
                    for doc in docs
                ]
            }
        )
    except Exception as e:
        return Result(value=f"Error embedding from file: {str(e)}")

def search_documents(query: str, top_k: int = 3) -> Result:
    """
    Search documents using semantic similarity.
    """
    try:
        results = doc_embedder.semantic_search(query, top_k)
        formatted_results = "\n\n".join([
            f"Document: {doc.metadata.get('title', 'Untitled')}\n"
            f"Content: {doc.text}\n"
            f"Metadata: {doc.metadata}"
            for doc in results
        ])
        return Result(
            value=f"Found {len(results)} relevant documents:\n\n{formatted_results}",
            context_variables={
                "last_search": {
                    "query": query,
                    "results": [doc.id for doc in results]
                }
            }
        )
    except Exception as e:
        return Result(value=f"Error searching documents: {str(e)}")

# Create the document embedding agent
embedding_agent = Agent(
    name="AI History Expert",
    model="gpt-4o",
    instructions="""You are an AI history expert that helps users explore and understand the history of artificial intelligence.

You have access to a knowledge base of AI-related documents that you can search and reference.

You can:
1. Embed new documents about AI history
2. Search existing documents to answer questions
3. Provide detailed explanations and context about AI developments

When searching or discussing AI history:
- Always cite specific events, dates, and developments from the documents
- Explain the significance of key milestones
- Connect historical events to modern AI developments
- Maintain a conversational yet informative tone

Example interactions:
- "Tell me about early AI development"
- "What happened during the AI winters?"
- "How has deep learning changed AI?"
- "What were the major breakthroughs in AI?"

First, I'll help you load our test document about AI history:
embed_from_file("test_documents/ai_history.json")

Then you can ask me questions about AI history, and I'll search the document to provide accurate, contextual answers.""",
    functions=[
        embed_document,
        embed_from_file,
        search_documents
    ]
)

# Initialize tangent client
client = tangent()

if __name__ == "__main__":
    from tangent.repl import run_tangent_loop
    
    print("\nAI History Expert Ready!")
    print("I'll start by loading our test document about AI history...")
    
    # First, embed our test document
    response = client.run(
        agent=embedding_agent,
        messages=[{"role": "user", "content": "Please load the AI history document."}],
        context_variables={},
        stream=False
    )
    
    print("\nYou can now ask questions about AI history! Try these examples:")
    print("1. When was the term 'Artificial Intelligence' first coined?")
    print("2. What happened in the AI winters?")
    print("3. What were the major AI breakthroughs in the 2010s?")
    print("\nStarting conversation...\n")
    
    # Run the interactive demo loop
    run_tangent_loop(embedding_agent, stream=True, debug=False)
