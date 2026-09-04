# post-graph-rag

[![PyPI version](https://img.shields.io/pypi/v/post-graph-rag.svg)](https://pypi.org/project/post-graph-rag/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Graph RAG with a memory of time — on the PostgreSQL you already run.**

`post-graph-rag` extracts entities and relations with an LLM, stores them as a property graph beside `pgvector` embeddings, and answers questions by fusing vector similarity, graph traversal and full-text search. What makes it different: **a later document can close an earlier fact**, so your model stops reporting that someone is both an ally and a rival.

No separate vector store. No graph engine to operate. One database, one consistency model, one backup — and transactions that span your graph *and* your application tables.

## 📈 Benchmarks

Paper: [arXiv:2608.24921](https://arxiv.org/abs/2608.24921).

### LongMemEval: long-horizon chat memory

On the full 500-question [LongMemEval](https://arxiv.org/abs/2410.10813) set — all six question types, nothing sampled — against the numbers Zep publish for Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)):

| | overall | multi-session | temporal | knowledge-update |
| :--- | ---: | ---: | ---: | ---: |
| **post-graph-rag** · `gemini-3.6-flash` | **94.0%** | **90.2%** | **96.2%** | **94.9%** |
| Zep/Graphiti · gpt-4o | 71.2% | 57.9% | 62.4% | 83.3% |
| Zep/Graphiti · gpt-4o-mini | 63.8% | 40.6% | 36.5% | 76.9% |
| Full-context baseline · gpt-4o | 60.2% | 44.3% | 45.1% | 78.2% |

*Qualifications, so you can weigh them yourself: Zep judge with GPT-4o, this uses a three-model majority panel with the answering model excluded from its own jury; the generation models differ in cost class and vintage in a direction that cannot be signed; one question of 500 is excluded because neither extraction prompt could turn that session into triples. The harness, frozen config and every failing case ship in the repo — see the [full write-up](https://crajah.github.io/post-graph-rag/), which also documents the improvements that were tested and rejected.*


### ECT-QA: the adversarial corpus

Chat memory is the easy register. [ECT-QA](https://arxiv.org/abs/2510.13590) is the hard one — earnings call transcripts, sixteen quarters per company, **every metric restated every quarter with only the date to tell the values apart.** A store that merely accumulates cannot answer these at all; it has four values for "gross margin" and no way to choose.

Scored under the protocol ECT-QA's own authors use — an LLM judge comparing element-wise, with a refusal counted separately from a wrong answer:

| | Correct ↑ |
| :--- | ---: |
| **post-graph-rag** · `gemini-3.6-flash` | **0.677** |
| TG-RAG *(published)* | 0.599 |
| GraphRAG *(published)* | 0.405 |
| LightRAG *(published)* | 0.406 |

A second judge from a different model family scores the same answers at 0.668 — agreement within a point, which matters more than either figure, since the usual objection to a judged rate is that it moves with the judge.

The breakdown is more useful than the total: **incorrect elements sit at 0.15**, and relative-time questions produce none at all — when this system commits to a figure, it is usually right. The gap to a higher score is *refusal*, evidence not retrieved rather than reasoning gone wrong.

*Same qualifications apply, plus two specific to this comparison: their judge model is not available on our router, and their verbatim rubric is truncated in the public HTML, so ours reproduces their described categories rather than their text. Their figures are on their corpus slice; ours is 78 questions over 6 companies. Read a few points of margin as approximate rather than decisive.*

---

## 🌟 Why `post-graph-rag`?

Traditional Vector RAG systems suffer from **"chunk isolation"**—they retrieve isolated text passages based purely on semantic similarity, missing higher-level relationships and cross-document entity connections. 

`post-graph-rag` solves this by building a **dual representation** inside PostgreSQL:
1. **Unstructured Vector Passages**: Full document chunks indexed with `pgvector` HNSW embeddings.
2. **Knowledge Graph Triples**: Extracted Subject-Predicate-Object entities connected by graph edges.
3. **Structured Document Metadata**: Rich metadata tracking (`source`, `category`, `collection`, `document`, `page`, `paragraph`, `space`).
4. **Application-Level Space Sub-grouping (`space`)**: Scopes indexing and vector similarity search to application-specific environments (e.g. `production`, `sandbox`, `staging`, `user_workspace`) within `{realm}` tenant partitions.

### Relationship quality

Extracted relations are **context-specific**, drawn from what the text actually
states:

```
(Zeus) --[is_king_of]--> (Olympian gods)
(Zeus) --[son_of]--> (Cronus)
(Zeus) --[married_to]--> (Hera)
(Zeus) --[defeated]--> (Titans)
```

Vague connectors (`relates_to`, `associated_with`, `connected_to`, …), self-loops
and blank endpoints are rejected at extraction time. A relation that reaches the
graph always says something specific about the pair it connects, and two entities
merely appearing near each other never produces an edge.

If the LLM cannot produce usable structure, indexing raises `ExtractionError`.
Placeholder edges are never invented as a fallback: once written they are
indistinguishable from genuine extracted structure.

Relations the text explicitly *denies* are stored with the positive predicate and
`negated: true`, rather than as an inverted predicate like
`did_not_have_relationship_with`. Traversal and synthesis can then exclude them
instead of reading them as assertions.

### Entity resolution

Entities are unique per `(realm, space, lower(name))`, enforced by a unique index,
and are additionally resolved through **aliases**. Extraction records every other
surface form it sees, so `Babbage`, `Charles Babbage` and `C. Babbage` converge on
one vertex; the fuller name becomes canonical and the rest become aliases. Pronouns
and relative references (`he`, `his father`, `the company`) are rejected outright —
they cannot resolve to a stable vertex.

The same entity mentioned in many documents is one vertex, which is what allows the
graph to connect chunks that share no vocabulary.

### Chunking and document context

`index_text()` chunks a document (with overlap, so relations spanning a boundary
survive) and threads a `DocumentContext` through the chunks — title, source, and
the canonical entity names found so far. Without it, every chunk after the first is
extracted blind and its pronouns become junk vertices.

Bring your own splitter by passing `chunker=` to `GraphRAG`, or use
`index_document()` directly with your own `DocumentContext`.

```python
rag = GraphRAG(config, chunker=my_splitter)
await rag.index_text(long_text, metadata=DocumentMetadata(document="babbage.txt"))
```

### Community summarisation

Corpus-level questions — "what are the main themes here?" — cannot be answered by
retrieving passages, because no single passage contains the answer. After indexing,
cluster the entity graph and summarise each cluster:

```python
await rag.index_text(doc_a, metadata=DocumentMetadata(document="a.txt"))
await rag.index_text(doc_b, metadata=DocumentMetadata(document="b.txt"))

await rag.build_communities()          # clusters + one LLM report per community

res = await rag.query("What are the main themes?", param=QueryParam(mode="global"))
print(res["retrieved_communities"])
```

Each report is stored as a vertex in `communities` **with its own embedding**, so
`global` and `hybrid` retrieval find themes by similarity rather than by
enumerating relations. Membership is recorded as `community_members` edges back to
the entities, so a report is always traceable to the subgraph it came from.

Communities are derived data: `build_communities()` replaces the previous
clustering for the space rather than accumulating stale clusters. Global mode
degrades to relation ranking when none have been built, so it never hard-fails.

Detection uses Leiden (`igraph` + `leidenalg`, installed by default), falling back
to deterministic label propagation if the native build is unavailable. Both are
deterministic — a randomised partition would produce a different graph on every
indexing run. Leiden is the default because partition balance matters: on the
evaluation corpus the largest community holds 35% of the graph under label
propagation against 17% under Leiden, and a community spanning a third of the
graph summarises everything rather than a theme. Supply your own with
`community_detector=`:

```python
rag = GraphRAG(config, community_detector=my_detector)   # (nodes, edges) -> {node: community_id}
```

### Repeated relations

The same triple extracted from several chunks is one edge whose `weight`
increments, not several edges. Weight then breaks ties when ranking relations.

---

## 🧭 Exploration Support (1.10.0)

Three engine calls for exploration-first consumers — structure, coverage,
change — with the agent loop staying yours:

```python
# Structure: an opt-in topic tree above the flat communities
rag = GraphRAG(RAGConfig(community_levels=2))
tree = await rag.get_community_tree()

# Coverage: where has retrieval never looked? (opt-in telemetry, hash-only)
frontier = await rag.least_explored_communities(k=5)
dark = await rag.dark_entities(limit=100)

# Change: what moved since the last poll, from belief time
delta = await rag.changes_since(watermark)          # counts only, one round trip
if not delta.empty:
    detail = await rag.changes_since(watermark, summary=False)
watermark = delta.as_of
```

Hierarchy levels nest by construction (recursive supergraph clustering);
level-filtered retrieval happens inside the vector search, not as a post-hoc
trim; deltas are exactly-once under clock skew via database-clock watermarks;
and re-indexing an unchanged document yields an empty delta. `community_levels`
defaults to 1 — existing behaviour is untouched.

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

## 🧪 Runnable Examples

Seven scripts in [`examples/`](examples/), each one capability, all runnable:

```bash
export OPENAI_API_KEY=...  OPENAI_API_BASE=...  POSTGRES_URI=...
cd examples && python 01_quickstart.py
```

| | shows |
| :--- | :--- |
| [`01_quickstart.py`](examples/01_quickstart.py) | Three documents, one question that needs all three |
| [`02_supersession.py`](examples/02_supersession.py) | A later filing closes an earlier fact; history stays queryable |
| [`03_bitemporal_audit.py`](examples/03_bitemporal_audit.py) | Reproduce what the system believed before a restatement |
| [`04_multi_tenant_spaces.py`](examples/04_multi_tenant_spaces.py) | Per-tenant isolation, plus deliberate cross-tenant views |
| [`05_incremental_and_deltas.py`](examples/05_incremental_and_deltas.py) | Idempotent re-indexing and change polling |
| [`06_communities_and_exploration.py`](examples/06_communities_and_exploration.py) | Topic tree, corpus themes, coverage gaps |
| [`07_retrieval_modes.py`](examples/07_retrieval_modes.py) | `local`, `global` and `mix` retrieval, same question |

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

`pgvector` is **required**, not optional. Without it the vertex tables are created
without embedding columns and every similarity search silently returns nothing.
`initialize()` raises `SchemaError` if it is missing rather than degrading.

```bash
# macOS
brew install pgvector

# Debian/Ubuntu (match your server version)
sudo apt install postgresql-17-pgvector
```

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
        api_key=os.environ["OPENAI_API_KEY"],     # Never hardcode credentials
        model="gemini-3.6-flash",                    # LLM model for extraction & synthesis
        embedding_model="gemini-embedding-001", # Embedding model
        embedding_dim=1536,                       # Must match the model's output width
        db_uri="postgresql://user:password@localhost:5432/postgres",
        realm="enterprise_kb",
        schema_per_realm=True                     # Give each tenant its own schema
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
    space="production",                        # Sub-grouping within the realm
    extra={"author": "Homer", "year": -700}    # Custom metadata key-value pairs
)

await rag.index_document(chunk_text, metadata=metadata)
```

### Design Rationale: Optional vs. Required
- **All metadata fields are optional** with default `None`. This allows seamless indexing of raw strings, short code snippets, webhooks, or unformatted text, while offering rich structural provenance tracking when indexing multi-page PDFs or categorized enterprise documents.

### Document identity: `source` and `document` together

Re-indexing **replaces** rather than appends, so two documents that resolve to the
same key are treated as one document seen twice — the second deletes the first.

The key is built from `source` **and** `document` together, so give at least one
of them a value that is unique per document:

```python
# Correct: source identifies this document
DocumentMetadata(source="/corpus/WDC-2022-q1.json", document="WDC-2022-q1")

# Wrong: a corpus name is not a document identity
DocumentMetadata(source="ect", document="WDC-2022-q1")   # was catastrophic before 1.8.0
```

Before 1.8.0 the key preferred `source` and ignored `document` entirely, so the
second form collapsed an entire corpus onto one key. An 80-transcript run kept
five transcripts and marked 92% of its relations dormant, with no error raised —
the only visible symptom was the system declining to answer questions whose
evidence had been deleted. Since 1.8.0 both parts contribute, so the second form
is merely untidy rather than destructive.

**Upgrading:** keys computed before 1.8.0 do not match the new scheme, so
re-indexing an existing document appends a copy instead of replacing it. Rebuild
realms indexed on the old scheme.

---

## ⚙️ Configuration Reference (`RAGConfig`)

`RAGConfig` can be configured explicitly or automatically loaded from environment variables:

| Option | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `api_base` | `OPENAI_API_BASE` | `http://localhost:4000/v1` | Base URL for OpenAI-compatible LLM endpoint |
| `api_key` | `OPENAI_API_KEY` | `EMPTY` | API key. `EMPTY` is the placeholder local servers accept |
| `model` | `RAG_MODEL` | `gemini-3.6-flash` | Primary LLM model for triple extraction & synthesis |
| `embedding_model` | `RAG_EMBEDDING_MODEL` | `gemini-embedding-001` | Model for vector embedding generation |
| `embedding_dim` | `RAG_EMBEDDING_DIM` | `1536` | Embedding width. Must match the model, and is fixed once tables exist |
| `db_uri` | `POSTGRES_URI` | `postgresql://localhost:5432/postgres` | PostgreSQL connection DSN |
| `realm` | `RAG_REALM` | `default` | Multi-tenant graph namespace |
| `space` | `RAG_SPACE` | `default` | Sub-grouping within a realm (`production`, `sandbox`, …) |
| `schema_per_realm` | `RAG_SCHEMA_PER_REALM` | `0` | Give each realm its own PostgreSQL schema. Recommended — see below |
| `embed_relations` | `RAG_EMBED_RELATIONS` | `1` | Embed relation edges so retrieval can find them by similarity as well as by traversal. Costs one embedding call per distinct triple at index time |
| `relation_seed_quota` | `RAG_RELATION_SEED_QUOTA` | `0.5` | Share of relation slots reserved for the similarity channel. `0` disables it |
| `allow_embedding_fallback` | `RAG_ALLOW_EMBEDDING_FALLBACK` | `0` | Use local/deterministic vectors when the embedding API fails |
| `fallback_models` | `RAG_FALLBACK_MODELS` | — | Comma-separated models to fail over to when the primary is rate-limited or out of credits |
| `max_retries` | `RAG_MAX_RETRIES` | `5` | Attempts per model before moving to the next |
| `gleaning_passes` | `RAG_GLEANING_PASSES` | `1` | Extra "what did you miss?" extraction passes. `0` halves LLM cost at the price of recall |
| `extraction_prompt` | — | `None` | Replace the extraction system prompt wholesale |
| `entity_types` | `RAG_ENTITY_TYPES` | library defaults | Preferred entity type list |
| `predicate_vocabulary` | `RAG_PREDICATE_VOCABULARY` | — | Preferred predicates; extracted ones are snapped onto this list |
| `predicate_aliases` | — | `{}` | Explicit synonym map, e.g. `{"collaborated_with": "worked_with"}` |
| `drop_negated_relations` | `RAG_DROP_NEGATED` | `0` | Discard relations the text says do *not* hold, instead of flagging them |
| `min_relation_confidence` | `RAG_MIN_RELATION_CONFIDENCE` | `0.0` | Drop relations below this extraction confidence |
| `chunk_chars` / `chunk_overlap_chars` | `RAG_CHUNK_CHARS` / `RAG_CHUNK_OVERLAP` | `2000` / `200` | Default chunker sizing |
| `expand_chunks_via_mentions` | `RAG_EXPAND_VIA_MENTIONS` | `1` | Retrieve chunks that mention a matched entity, not only chunks matching the query vector |
| `context_entity_limit` | `RAG_CONTEXT_ENTITY_LIMIT` | `40` | Canonical names carried forward as extraction context |
| `community_min_size` | `RAG_COMMUNITY_MIN_SIZE` | `3` | Smallest cluster worth summarising |
| `community_resolution` | `RAG_COMMUNITY_RESOLUTION` | `1.0` | Higher yields more, smaller communities (Leiden only) |
| `max_communities` | `RAG_MAX_COMMUNITIES` | `64` | Cap per build; each community costs one LLM call |
| `community_report_prompt` | — | `None` | Replace the community report prompt |
| `negated_relation_weight` | `RAG_NEGATED_RELATION_WEIGHT` | `0.3` | Clustering weight for denied relations |

Environment variables are read when a `RAGConfig` is constructed, not at import time.

### `schema_per_realm`

Off by default for backwards compatibility, but recommended for new deployments.
With it off, every realm shares one physical set of tables filtered by a `realm`
column — so the first realm to create `entities` fixes the embedding column width
for all of them, and a second realm with a different `embedding_dim` cannot work.

### `allow_embedding_fallback`

Off by default. Fallback vectors are not comparable with API embeddings, so mixing
them into the same table corrupts retrieval rather than degrading it. With it off,
an embedding failure raises `EmbeddingError`.

---

## 📖 API Reference

### `GraphRAG`
The main orchestrator class for indexing and querying.

- `await initialize()`: Connects to PostgreSQL and creates the graph tables (`documents`, `entities`, `relations`, `doc_mentions`). Raises `SchemaError` if pgvector is unavailable or an existing table's embedding width disagrees with `embedding_dim`.
- `await index_document(text, metadata=None, space=None) -> Dict[str, Any]`: Embeds the chunk, extracts entities/triples via the LLM, resolves entities by name, and writes vertices, `relations` and `doc_mentions` edges. Raises rather than writing placeholder structure if extraction fails. Returns counts plus `document_id` and `metadata`.
- `await query(question, param=None, top_k=None)`: Retrieves and synthesizes an answer. Returns `question`, `answer`, `mode`, `keywords`, `retrieved_documents`, `retrieved_entities`, `retrieved_graph_triples`, `references`. With `QueryParam(stream=True)` returns an async iterator of content chunks instead.
- `await query_data(question, param=None) -> Dict[str, Any]`: Structured retrieval with no synthesis — returns `entities`, `relationships`, `chunks`, `references`.
- `await close()`: Closes database connection pools.

### `QueryParam`

- `mode`: one of `mix`, `local`, `global`, `hybrid`, `naive`, `bypass`. An unknown mode raises `ValueError`.
- `top_k`, `max_total_tokens`, `max_entity_tokens`, `max_relation_tokens`, `response_type`
  - The three token budgets default to `None` — **unlimited** — as of 1.11.1: everything retrieved reaches the model. Set an integer to cap context for cost or for a model with a small window.
- `stream`: return an async iterator of tokens instead of a dict.
- `only_need_context`: return retrieval output without calling the LLM.
- `space`: restrict retrieval to one space; `__all__` queries across all spaces.
- `conversation_history`, `hl_keywords`, `ll_keywords`: supply keywords to skip extraction.

### Errors

All inherit from `RAGError`, so failures surface instead of degrading into
irrelevant results:

- `SchemaError` — pgvector missing, or embedding width mismatch.
- `EmbeddingError` — embedding request failed, or returned the wrong width.
- `LLMError` — completion or streaming call failed.
- `ExtractionError` — the LLM returned no usable entities or triples.

### `DocumentMetadata`
Data container for structured document metadata.

- `source: Optional[str]`: Document URL, path, or origin.
- `category: Optional[str]`: Document category or domain.
- `collection: Optional[str]`: Document collection or folder.
- `document: Optional[str]`: File title or filename.
- `page: Optional[int]`: 1-based page number.
- `paragraph: Optional[int]`: 1-based paragraph index.
- `space: Optional[str]`: Sub-grouping space to index into.
- `extra: Dict[str, Any]`: Custom user metadata.
- `to_dict() -> Dict[str, Any]`: Serializes non-None fields to dictionary representation.
- `from_dict(data: Dict[str, Any]) -> DocumentMetadata`: Deserializes dictionary data.

### `RAGGraphStore`
Database layer wrapping `post-graph`.

- `add_document(text, embedding, metadata, space=None)`: Inserts a document vertex into the `documents` table.
- `upsert_entity(name, entity_type, description, embedding, space=None)`: Upserts by canonical name within `(realm, space)`, so an entity mentioned in many documents is a single vertex. A bare `Concept` stub never overwrites a richer type or description.
- `find_entity_by_name(name, space=None)`: Resolve an entity vertex by name.
- `add_relation(from_entity, to_entity, relation_type, description, space=None, embedding=None)`: Directed relation edge.
- `add_doc_mention(doc_vertex, entity_vertex, space=None)`: Links a chunk to an entity it mentions.
- `search_similar_entities(query_vec, top_k, space=None)` / `search_similar_documents(...)`: pgvector HNSW similarity search.
- `search_similar_relations(query_vec, top_k, space=None)`: Semantic search over relation edges. Returns `[]` unless `embed_relations` is enabled.
- `get_neighbors(entity_id, space=None)`: 1-hop outgoing relations, scoped to `space`.
- `get_all_relations(limit, space=None)`: Relations with their endpoint vertices.

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

## 🧪 Testing

```bash
pip install -e ".[test]"
createdb postgres && psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
POSTGRES_TEST_URI="postgresql://localhost:5432/postgres" pytest
```

DB-backed tests create a disposable schema per test realm and drop it afterwards.
They skip automatically when PostgreSQL with pgvector is not reachable.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Developed by **Chandan Rajah** (<chandan.rajah@gmail.com>).
