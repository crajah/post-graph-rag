# Free-Tier Cloud Quickstart: Neon & Supabase (60 Seconds)

If you do not run PostgreSQL with `pgvector` locally on macOS or Windows, both **Neon** and **Supabase** ship PostgreSQL 16/17 with `pgvector` pre-installed on their free tiers. Provisioning a database takes under a minute and works with both the **`post-graph-rag-mcp`** IDE server (Python 3.11–3.14, zero LLM API keys required) and the **`post-graph-rag`** Python library.

---

## 1. Provision a Free PostgreSQL + `pgvector` Database

### Option A — Neon (Serverless PostgreSQL)

1. Create a free project at [neon.tech](https://neon.tech) (select **PostgreSQL 16** or **17**).
2. Open the **SQL Editor** in the Neon console and enable `pgvector` once:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Copy your connection string from **Connection Details**:
   ```text
   postgresql://<user>:<password>@ep-example-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

> **Direct vs `-pooler` endpoints on Neon:**
> - Using Neon's **direct** endpoint (`ep-example-123456...`) works with default settings.
> - Using Neon's **pooled** endpoint (`ep-example-123456-pooler...`, which runs PgBouncer in transaction mode) rotates backend connections between transactions — pass `RAG_STATEMENT_CACHE_SIZE=0` (`statement_cache_size=0` in `RAGConfig`) so `asyncpg` never hits `DuplicatePreparedStatementError`.

### Option B — Supabase (Managed PostgreSQL)

1. Create a free project at [supabase.com](https://supabase.com).
2. Open the **SQL Editor** (or **Database -> Extensions**) and enable `pgvector` in `public`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
   ```
3. Copy your connection string from **Project Settings -> Database -> Connection string (URI)**:
   - **Session pooler (port `5432`)**:
     ```text
     postgresql://postgres.<project-ref>:<password>@aws-0-us-east-1.pooler.supabase.com:5432/postgres
     ```
   - **Transaction pooler (port `6543`)**: also set `RAG_STATEMENT_CACHE_SIZE=0` (`statement_cache_size=0` in `RAGConfig`).

---

## 2. Connect Claude Code or Cursor (`post-graph-rag-mcp`)

With an external `POSTGRES_URI`, `post-graph-rag-mcp` needs no `--python` version pin (runs on Python 3.11–3.14) and computes 384-dimensional embeddings locally via `fastembed` (`BAAI/bge-small-en-v1.5`) — **no `OPENAI_API_KEY` required**.

### Claude Code (One Command)

```bash
claude mcp add post-graph-rag \
  -e POSTGRES_URI="postgresql://..." \
  -e RAG_STATEMENT_CACHE_SIZE="0" \
  -- uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp
```

### Cursor / Windsurf (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "post-graph-rag": {
      "command": "uvx",
      "args": [
        "--from",
        "post-graph-rag-mcp[local-embed]",
        "post-graph-rag-mcp"
      ],
      "env": {
        "POSTGRES_URI": "postgresql://...",
        "RAG_STATEMENT_CACHE_SIZE": "0"
      }
    }
  }
}
```

### Verify Connectivity in 2 Seconds

```bash
POSTGRES_URI="postgresql://..." RAG_STATEMENT_CACHE_SIZE="0" \
  uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp doctor
```

---

## 3. Connect from Python (`post-graph-rag`)

Install the core library and point `RAGConfig` at the same `POSTGRES_URI` plus any OpenAI-compatible router (`OpenAI`, `OpenRouter`, `LiteLLM`, `vLLM`, `Ollama`):

```bash
pip install post-graph-rag
```

```python
import asyncio
import os
from post_graph_rag import GraphRAG, RAGConfig


async def main() -> None:
    rag = GraphRAG(RAGConfig(
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.environ.get("RAG_EMBEDDING_DIM", "1536")),
        db_uri=os.environ["POSTGRES_URI"],
        realm="quickstart_demo",
        schema_per_realm=True,
        statement_cache_size=0,  # safe on both direct connections and transaction poolers
        exclusive_predicate_groups=[{"appointed_cfo_of", "former_cfo_of"}],
        functional_predicates={"cfo_of"},
    ))
    await rag.initialize()

    await rag.index_document(
        "Helena Voss is CFO of Calder Industries.",
        metadata={"source": "2019-10k.txt", "document": "2019-10k"},
    )
    await rag.index_document(
        "Priya Nair has been appointed CFO of Calder Industries, succeeding Helena Voss.",
        metadata={"source": "2023-10k.txt", "document": "2023-10k"},
    )

    result = await rag.query("Who is the CFO of Calder Industries?")
    print(result["answer"])
    await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Run the Full Example Suite Against Neon or Supabase

All seven scripts in [`examples/`](./) read `POSTGRES_URI`, `OPENAI_API_BASE`, and `OPENAI_API_KEY`:

```bash
export POSTGRES_URI="postgresql://..."
export OPENAI_API_BASE="https://your-router/v1"
export OPENAI_API_KEY="..."
export RAG_STATEMENT_CACHE_SIZE="0"

cd examples
python 01_quickstart.py
python 02_supersession.py
python 03_bitemporal_audit.py
```
