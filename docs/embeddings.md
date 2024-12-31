Below is a **complete and self-contained explanation** of how the tangent Python library **uses the `text-embedding-3-large` (and similarly `text-embedding-3-small`) embeddings models**, based **entirely** on the provided documentation. This guide will detail **all the ways** these embeddings can be utilized in a tangent-based project, along with **every step** present in the example codebases that show how these embeddings are created, stored, and retrieved.

---

## 1. Where `text-embedding-3-large` Appears in tangent

In the **examples** directory, you can see references to the **embedding model** parameter set to `"text-embedding-3-large"`. Specifically:

- **`examples/customer_service_streaming/prep_data.py`**  
- **`examples/customer_service_streaming/configs/tools/query_docs/handler.py`**  
- **`examples/support_bot/prep_data.py`**  

In each of these, **OpenAI’s** embeddings endpoint is used to transform text into high-dimensional vectors. The code calls:

```python
client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=x['text']
)
```

Here, `EMBEDDING_MODEL` is the string `'text-embedding-3-large'` or `'text-embedding-3-small'`.

---

## 2. Overview: What Happens With Embeddings in tangent Projects

1. **Data is Preprocessed**: In the examples, JSON articles or documents are loaded from disk, and each article’s text is run through the embedding model (`text-embedding-3-large` or `text-embedding-3-small`) to create vector embeddings.  
2. **Vectors are Stored in Qdrant**: tangent can integrate with **Qdrant**—a vector database. The code indexes these embeddings along with their metadata (title, text, etc.) into a Qdrant collection.  
3. **Querying by Embeddings**: When an end user poses a question, tangent calls `query_qdrant`, which again uses the embedding model to transform the user’s query text into a vector, then performs a nearest-neighbor search in Qdrant. The best matching articles are returned.  
4. **Use in Agents**: Various tangent agents or tools (e.g., `query_docs`) can rely on this embedding-based search to help answer user questions about the stored data.

---

## 3. Creating Embeddings During Data “Prep” Stage

### Example: `examples/customer_service_streaming/prep_data.py`

```python
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-large"

article_list = os.listdir('data')
articles = []

# Load each article JSON
for x in article_list:
    article_path = 'data/' + x
    f = open(article_path)
    data = json.load(f)
    articles.append(data)
    f.close()

# Generate embeddings
for i, x in enumerate(articles):
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=x['text'])
        articles[i].update({"embedding": embedding.data[0].embedding})
    except Exception as e:
        print(x['title'])
        print(e)
```

**Step-by-Step**:

1. The code loops over JSON files in a `data/` directory, each containing article text.  
2. Each article’s text is passed to `client.embeddings.create(...)`, specifying `"text-embedding-3-large"` as the model.  
3. The resulting embedding (a list of floating-point numbers) is added to the article dictionary under the key `"embedding"`.

### 3.1 Inserting Embeddings Into Qdrant

Immediately after generating each embedding, the example code stores them in a Qdrant vector database:

```python
qdrant.upsert(
    collection_name=collection_name,
    points=[
        rest.PointStruct(
            id=k,
            vector={'article': v['embedding']},
            payload=v.to_dict(),
        )
        for k, v in article_df.iterrows()
    ],
)
```

- A new Qdrant collection is created or re-created to hold these vectors.  
- Each article’s embedding becomes the `'article'` vector.  
- All the textual metadata (title, text, etc.) goes into `payload`.

By doing this, an entire corpus of documents is now searchable via vector embeddings.

---

## 4. Using Embeddings for Document Retrieval

### Example: `examples/customer_service_streaming/configs/tools/query_docs/handler.py`

```python
EMBEDDING_MODEL = 'text-embedding-3-large'

def query_docs(query):
    print(f'Searching knowledge base with query: {query}')
    query_results = query_qdrant(query, collection_name=collection_name)
    ...
```

And inside **`query_qdrant`**:

```python
def query_qdrant(query, collection_name, vector_name='article', top_k=5):
    # Convert user query to embedding
    embedded_query = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    ).data[0].embedding

    # Perform vector search in Qdrant
    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
    )
    return query_results
```

Here’s the **flow**:

1. **User** interacts with a tangent agent that calls `query_docs(...)`.  
2. The function **re-embeds** the user’s query into a vector using `"text-embedding-3-large"`.  
3. That vector is used to **search** the Qdrant index, retrieving the top matching articles.  
4. The matching article(s) are returned to the agent, which can integrate that content into a response.

---

## 5. Additional Notes on Usage With tangent

- **Agent Tools**: In `assistant.json` or in your `Agent` definition, you may define a function like `query_docs`. That function is recognized as a “tool” by the tangent framework.  
- **Parallel or Single Tool Calls**: Agents in tangent can either call multiple tools in parallel or call them sequentially (depending on the `parallel_tool_calls` setting). This includes the embedding-based tool if the conversation demands it.  
- **Combining With Other Tools**: Once data is retrieved via embeddings, you might chain it with a second tool, such as `send_email`, to mail the user’s requested information.  
- **Scaling**: Whether you use `"text-embedding-3-large"` or `"text-embedding-3-small"`, the code pattern remains the same. You set `EMBEDDING_MODEL` in your Python script, then call `client.embeddings.create(model=EMBEDDING_MODEL, input=...)`.

---

## 6. Summary of Steps

1. **Set EMBEDDING_MODEL** (e.g., `"text-embedding-3-large"`) in your script.  
2. **Load your documents** from JSON files or other data sources.  
3. **Generate embeddings** for each document by calling:  
   ```python
   embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=document_text)
   ```
4. **Store** those embeddings into a vector database (like Qdrant) along with the documents.  
5. **At query time**, embed the user’s query text with the **same** model, and pass that vector to Qdrant for nearest-neighbor search.  
6. **Return** the most relevant docs to the user or agent.  

Hence, tangent’s usage of `text-embedding-3-large` (or `text-embedding-3-small`) centers on:

- **Preprocessing** text into vectors  
- **Storing** those vectors in Qdrant  
- **Searching** that vector space to find relevant information for the user  

Everything is orchestrated seamlessly within the multi-agent environment that tangent provides.

---

**That’s all, based entirely on the provided tangent documentation.**