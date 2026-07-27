# post-graph-rag

[![PyPI version](https://img.shields.io/pypi/v/post-graph-rag.svg)](https://pypi.org/project/post-graph-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Production-Grade, High-Performance Knowledge Graph RAG Engine Native to PostgreSQL.**

`post-graph-rag` seamlessly combines **automated LLM-based entity & triple extraction**, **vector similarity search via `pgvector`**, and **graph relationship traversal** directly on PostgreSQL using the [`post-graph`](https://pypi.org/project/post-graph/) graph database library.

It connects to **any OpenAI-compatible API** (LiteLLM, vLLM, Ollama, DeepSeek, OpenAI) for zero-shot domain-agnostic knowledge extraction, structured document metadata tracking, and context-aware answer synthesis.

---

## 🌟 Why `post-graph-rag`?

Traditional Vector RAG systems suffer from **"chunk isolation"**—they retrieve isolated text passages based purely on semantic similarity, missing higher-level relationships and cross-document entity connections. 

`post-graph-rag` solves this by building a **dual representation** inside PostgreSQL:
1. **Unstructured Vector Passages**: Full document chunks indexed with `pgvector` HNSW embeddings.
2. **Knowledge Graph Triples**: Extracted Subject-Predicate-Object entities connected by graph edges.
3. **Structured Document Metadata**: Rich metadata tracking (`source`, `category`, `collection`, `document`, `page`, `paragraph`, `space`).
4. **Application-Level Space Sub-grouping (`space`)**: Scopes indexing and vector similarity search to application-specific environments (e.g. `production`, `sandbox`, `staging`, `user_workspace`) within `{realm}` tenant partitions.

---

## 🏗️ Architecture Workflow

```mermaid
graph TD
    subgraph INDEXING ["1. Knowledge Graph & Vector Indexing"]
        A[Document Text + Metadata] --> B[Embedding Service]
        A --> C[LLM GraphExtractor]
        
        B -->|Vectors| D[post-graph Store]
        C -->|Entities & Triples| D
        
        D --> E[(PostgreSQL + pgvector)]
        E -->|Tables| E1[documents]
        E -->|Tables| E2[entities]
        E -->|Edges| E3[relations]
        E -->|Edges| E4[doc_mentions]
    end

    subgraph RETRIEVAL ["2. Hybrid Retrieval & Synthesis"]
        Q[User Question] --> R[GraphRAG Query Engine]
        R -->|Embedding| S[pgvector Similarity Search]
        E1 & E2 -->|Top-K Passages & Entities| S
        S --> T[1-Hop Graph Relationship Traversal]
        E3 -->|Subject-Predicate-Object| T
        
        S & T --> U[LLM Answer Synthesis]
        U --> V[Final Answer + Citations + Graph Triples]
    end
```

---

## 📦 Installation

Install `post-graph-rag` via `pip` or `uv`:

```bash
pip install post-graph-rag
```

Or using `uv`:

```bash
uv add post-graph-rag
```

### PostgreSQL Requirements
Ensure PostgreSQL is running with the `pgvector` extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🚀 Quick Start

### 1. Basic Indexing & Querying

```python
import asyncio
from post_graph_rag import GraphRAG, RAGConfig, DocumentMetadata

async def main():
    # 1. Configure GraphRAG engine
    config = RAGConfig(
        api_base="http://localhost:4000/v1",       # OpenAI-compatible router endpoint
        api_key="BEVZ-6L81-OZ8Y",                 # Master or OpenAI API Key
        model="DeepSeek-V3.2",                    # LLM model for extraction & synthesis
        embedding_model="text-embedding-3-small", # Embedding model
        embedding_dim=1536,                       # Vector dimensionality
        db_uri="postgresql://user:password@localhost:5432/postgres",
        realm="enterprise_kb"
    )

    rag = GraphRAG(config)
    
    # 2. Connect & initialize PostgreSQL graph schema
    await rag.initialize()

    # 3. Index unstructured documents
    doc_text = (
        "Zeus is the king of the Olympian gods, ruling sky and thunder from Mount Olympus. "
        "He is the son of Cronus and Rhea, and married to Hera. "
        "Zeus defeated the Titans in the Titanomachy to establish his rule."
    )
    
    result = await rag.index_document(doc_text, metadata={"source": "greek_mythology.txt"})
    print(f"Indexed document {result['document_id']}: Extracted {result['entities_extracted']} entities.")

    # 4. Perform Hybrid RAG Query
    response = await rag.query("Who are the parents of Zeus and what did he defeat?")
    
    print("\n=== SYNTHESIZED ANSWER ===")
    print(response["answer"])

    print("\n=== RETRIEVED GRAPH TRIPLES ===")
    for triple in response["retrieved_graph_triples"]:
        print(f"  - {triple}")

    # 5. Clean up
    await rag.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📋 Document Metadata (`DocumentMetadata`)

`post-graph-rag` includes structured document metadata tracking via the `DocumentMetadata` model:

```python
from post_graph_rag import DocumentMetadata

metadata = DocumentMetadata(
    source="https://mythology.org/zeus.html",  # Document origin (URL, filepath, API)
    category="greek_mythology",               # Document category/topic
    collection="olympian_deities",             # Collection namespace
    document="zeus_overview.pdf",              # Title or filename
    page=1,                                    # 1-based page number
    paragraph=2,                               # 1-based paragraph index
    extra={"author": "Homer", "year": -700}    # Custom metadata key-value pairs
)

await rag.index_document(chunk_text, metadata=metadata)
```

### Design Rationale: Optional vs. Required
- **All metadata fields are optional** with default `None`. This allows seamless indexing of raw strings, short code snippets, webhooks, or unformatted text, while offering rich structural provenance tracking when indexing multi-page PDFs or categorized enterprise documents.

---

## ⚙️ Configuration Reference (`RAGConfig`)

`RAGConfig` can be configured explicitly or automatically loaded from environment variables:

| Option | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `api_base` | `OPENAI_API_BASE` | `http://localhost:4000/v1` | Base URL for OpenAI-compatible LLM endpoint |
| `api_key` | `OPENAI_API_KEY` | `BEVZ-6L81-OZ8Y` | API Key for authorization |
| `model` | `RAG_MODEL` | `DeepSeek-V3.2` | Primary LLM model for triple extraction & synthesis |
| `embedding_model` | `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for vector embedding generation |
| `embedding_dim` | `RAG_EMBEDDING_DIM` | `1536` | Dimensionality of embedding vectors |
| `db_uri` | `POSTGRES_URI` | `postgresql://crajah@localhost:5432/postgres` | PostgreSQL connection DSN |
| `realm` | `RAG_REALM` | `default` | Multi-tenant graph namespace |

---

## 📖 API Reference

### `GraphRAG`
The main orchestrator class for indexing and querying.

- `await initialize()`: Connects to PostgreSQL and creates necessary graph tables (`documents`, `entities`, `relations`, `doc_mentions`).
- `await index_document(text: str, metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None) -> Dict[str, Any]`: Computes document embeddings, extracts entity/triple structures via LLM, and persists graph nodes/edges into PostgreSQL.
- `await query(question: str, top_k: int = 5) -> Dict[str, Any]`: Executes hybrid vector similarity search over documents and entities, traverses 1-hop graph relationship edges, and synthesizes a comprehensive answer. Returns dictionary with `question`, `answer`, `retrieved_documents`, `retrieved_entities`, and `retrieved_graph_triples`.
- `await close()`: Closes database connection pools.

### `DocumentMetadata`
Data container for structured document metadata.

- `source: Optional[str]`: Document URL, path, or origin.
- `category: Optional[str]`: Document category or domain.
- `collection: Optional[str]`: Document collection or folder.
- `document: Optional[str]`: File title or filename.
- `page: Optional[int]`: 1-based page number.
- `paragraph: Optional[int]`: 1-based paragraph index.
- `extra: Dict[str, Any]`: Custom user metadata.
- `to_dict() -> Dict[str, Any]`: Serializes non-None fields to dictionary representation.
- `from_dict(data: Dict[str, Any]) -> DocumentMetadata`: Deserializes dictionary data.

### `RAGGraphStore`
Database layer wrapping `post-graph`.

- `add_document(text, embedding, metadata)`: Inserts a document vertex into the `documents` table.
- `upsert_entity(name, entity_type, description, embedding)`: Upserts an entity vertex into the `entities` table.
- `add_relation(from_entity, to_entity, relation_type, description)`: Connects entity vertices with a directed relation edge.
- `search_similar_entities(query_vec, top_k)`: Executes `pgvector` HNSW similarity search over `entities`.
- `search_similar_documents(query_vec, top_k)`: Executes `pgvector` HNSW similarity search over `documents`.

---

## 🗄️ PostgreSQL Database Schema

`post-graph-rag` automatically provisions and manages the following graph schema in PostgreSQL powered by `post-graph`:

| Table Name | Type | Key Columns | Description |
| :--- | :--- | :--- | :--- |
| `{realm}_documents` | Vertex Table | `id`, `payload`, `embedding` (`vector`) | Stores raw text chunks and `DocumentMetadata` payloads |
| `{realm}_entities` | Vertex Table | `id`, `payload`, `embedding` (`vector`) | Canonical entity nodes (`name`, `type`, `description`) |
| `{realm}_relations` | Edge Table | `from_id`, `to_id`, `relation_type`, `payload` | Directed edges representing entity-to-entity triples |
| `{realm}_doc_mentions` | Edge Table | `from_id`, `to_id`, `relation_type` | Directed edges connecting document chunks to mentioned entities |
| `{table}_audit` | Audit Table | `audit_id`, `action`, `changed_by`, `changed_at` | Automatic shadow audit logging for all graph mutations |
| `{table}_data` | History Table | `data_id`, `payload`, `timestamp`, `embedding` | Append-only historical records for vertices and edges |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed by **Chandan Rajah** (<chandan.rajah@gmail.com>).
