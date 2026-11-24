# post-graph-rag-mcp

[![PyPI version](https://img.shields.io/pypi/v/post-graph-rag-mcp.svg)](https://pypi.org/project/post-graph-rag-mcp/)
[![Core Engine](https://img.shields.io/pypi/v/post-graph-rag.svg?label=post-graph-rag)](https://pypi.org/project/post-graph-rag/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 16+](https://img.shields.io/badge/PostgreSQL-16%20%7C%2017%20%7C%2018-blue.svg)](https://www.postgresql.org/)

**Bi-temporal Graph RAG memory for Claude Code, Cursor, Windsurf, and MCP clients — on PostgreSQL + `pgvector`.**

Standard agent memory stores only accumulate: migrate from Clerk to BetterAuth or appoint a new CFO, and cosine search keeps serving the retired decision alongside the current one. `post-graph-rag-mcp` wraps [`post-graph-rag`](https://github.com/crajah/post-graph-rag) (`94.0%` on [LongMemEval](https://arxiv.org/abs/2410.10813), `0.807` on [ECT-QA](https://arxiv.org/abs/2510.13590) — see the [core README](https://github.com/crajah/post-graph-rag#benchmarks) for benchmark qualifications; paper: [arXiv:2608.24921](https://arxiv.org/abs/2608.24921)) as a Model Context Protocol server (`stdio` and Streamable HTTP) where **later assertions deterministically close superseded ones** while keeping both **valid time** (`as_of`) and **belief time** (`as_believed_at`) queryable for audit.

No Neo4j. No separate vector database. No sidecar LLM API key required on the interactive IDE path.

---

## Install in 20 Seconds

### Tier 1 — Zero-Config Local (`uvx`, Embedded Postgres + Local ONNX Embeddings)

Bundles a private PostgreSQL 16.2 + `pgvector` 0.6.2 instance (`pgserver` in `~/.local/share/post-graph-rag-mcp/pgdata`, shared safely across multiple IDE windows via file locking) and in-process `fastembed` (`BAAI/bge-small-en-v1.5`, 384d). **Zero Docker, zero cloud signup, zero API keys.**

> **Note on `--python 3.12`:** `pgserver v0.1.4` publishes prebuilt wheels up to CPython 3.12 (`cp39–cp312`). Passing `--python 3.12` lets `uvx` fetch that wheel automatically even if your system default Python is 3.13+.

**Claude Code:**

```bash
claude mcp add post-graph-rag -- uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp
```

**Cursor / Windsurf / Claude Desktop (`.cursor/mcp.json` or `claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "post-graph-rag": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "post-graph-rag-mcp[embedded,local-embed]",
        "post-graph-rag-mcp"
      ]
    }
  }
}
```

### Tier 2 — Existing PostgreSQL / Docker / Neon / Supabase (Python 3.11–3.14)

Point at any PostgreSQL 16–18 database with `pgvector` — no `--python` pin required. Need a free cloud database in 60 seconds? Follow the [Neon & Supabase Quickstart](https://github.com/crajah/post-graph-rag/blob/main/examples/00_neon_supabase_quickstart.md).

**Claude Code:**

```bash
claude mcp add post-graph-rag \
  -e POSTGRES_URI="postgresql://localhost:5432/postgres" \
  -- uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp
```

**Cursor / Windsurf / Claude Desktop:**

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
        "POSTGRES_URI": "postgresql://localhost:5432/postgres"
      }
    }
  }
}
```

### Tier 3 — Shared Team Server (`Streamable HTTP` + Docker Compose)

Run one shared bi-temporal project memory for an entire engineering team (`pgvector/pgvector:pg17` + non-root `python:3.13-slim` Streamable HTTP container on port `8080`):

```bash
git clone https://github.com/crajah/post-graph-rag.git
cd post-graph-rag/post-graph-rag-mcp
docker compose up -d --build
```

Connect MCP clients to `http://localhost:8080/mcp`. Set `OPENAI_API_KEY` (and optional `OPENAI_API_BASE` for LiteLLM, OpenRouter, vLLM, or Ollama) in the environment to enable autonomous multi-chunk graph extraction on background `ingest_documents` runs.

---

## Two Operating Modes

Inside an IDE that is already running a frontier coding model, forcing users to configure a second chat-LLM API key just to extract triples is pure friction. `post-graph-rag-mcp` avoids it cleanly:

| Capability | Mode A: Zero-Key Host-Agent Mode *(Default)* | Mode B: Autonomous Router Mode *(Set `OPENAI_API_KEY`)* |
| :--- | :--- | :--- |
| **Triple extraction on `remember`** | Host IDE model passes `triples` + `entities` directly in the tool call (`0` sidecar LLM calls) | Host triples if passed; otherwise `GraphExtractor` extracts & gleans via `RAG_EXTRACTION_MODEL` |
| **Embeddings** | Local ONNX via `fastembed` (`BAAI/bge-small-en-v1.5`, 384d, ~30 MB, offline after first pull) | Local `fastembed` by default, or `/v1/embeddings` when `MCP_EMBEDDING_BACKEND=openai` |
| **Supersession** | Functional predicates (`uses_auth`, `deployed_on`, …), exclusive predicate groups, polarity flips, and host-assisted `active_relations_on_subjects` -> `supersedes_edge_ids` | All deterministic passes **plus** optional `RAG_CONTRADICTION_DETECTION=1` LLM residual checks |
| **Retrieval (`recall` / `recall_as_of`)** | `GraphRAG.query_data()` with host keywords or deterministic lexical fallback (`0` chat-LLM calls) | Structured `query_data()` by default, or full `GraphRAG.query()` synthesis when `synthesize=True` |
| **Directory sync (`ingest_documents`)** | Client MCP sampling if supported; otherwise indexes chunk vectors + YAML frontmatter triples (`chunks_and_frontmatter`) | Full multi-chunk `GraphRAG.index_text()` with `DocumentContext` coreference threading (`llm_graph`) |

Each realm records its `(embedding_model, embedding_dim)` fingerprint in PostgreSQL (`schema_per_realm=True` by default) and raises `SchemaError` immediately if a later session attempts to mix incompatible vector spaces into an existing realm.

---

## Complete MCP Surface

### 7 Tools

| Tool | Purpose |
| :--- | :--- |
| `remember` | Persist a decision, convention, or fact as an auditable passage + graph triples. Supports `functional=True` per triple, `supersedes_edge_ids` for explicit retraction, and returns `superseded_relations` plus `active_relations_on_subjects` so the IDE model catches synonym drift in zero extra LLM calls. |
| `recall` | Hybrid current-truth retrieval fusing recursive CTE graph walks, `pgvector` HNSW on chunks and relation edges, and PostgreSQL `tsvector` GIN search via Reciprocal Rank Fusion (RRF). Superseded and negated assertions are excluded automatically. |
| `recall_as_of` | Bi-temporal time-travel & audit query along **valid time** (`as_of="2023-06-15"` — what held in the world on that date) and/or **belief time** (`as_believed_at="2026-11-01T12:00:00Z"` — what this store knew before a later restatement arrived). |
| `trace_history` | Bidirectional (`direction="both"`) lineage query returning `current_relations` and `superseded_relations` in chronological order — answers both *"What auth provider does AuthService use now vs before?"* and *"Where was Clerk used?"* |
| `ingest_documents` | Idempotent Markdown / ADR directory sync (`order_by="filename" \| "mtime" \| "git_first_commit"`), sandboxed to `MCP_ALLOWED_ROOTS`, skipping unchanged files in `O(1)` via content hash. |
| `changes_since` | Database-clock watermark delta feed (`CorpusDelta`) showing entities, relations, and documents added or superseded since a prior watermark without client clock skew. |
| `configure_memory_rules` | Inspect or extend active `functional_predicates` and `exclusive_predicate_groups` (`engineering` or `general_memory` presets), and inspect graph coverage, dark entities, and document stats. |

### 3 Resources

- `memory://status` — Active `realm`, `space`, embedding backend & dimension, document/entity counts, and current database watermark.
- `memory://rules` — Active preset, `functional_predicates`, `exclusive_predicate_groups`, `predicate_aliases`, and `predicate_vocabulary`.
- `memory://communities` — Hierarchical Leiden community summary tree (`get_community_tree()`).

### 3 Prompts

- `remember-architecture-decision` — Structured workflow for capturing a design decision or migration with `functional=True` predicates and checking `active_relations_on_subjects`.
- `audit-belief-state` — Side-by-side comparison of current state (`recall`), point-in-time belief (`recall_as_of`), and full supersession lineage (`trace_history`).
- `sync-adr-folder` — Guides an IDE agent through ordered ADR / documentation ingestion (`0001-*.md` -> `0099-*.md`) so later ADRs supersede earlier ones cleanly.

---

## Built-in Supersession Presets

- **`engineering` *(default)***
  - **Functional (single-valued per subject):** `uses_auth`, `uses_database`, `uses_framework`, `uses_orm`, `uses_queue`, `uses_package_manager`, `uses_test_runner`, `uses_bundler`, `uses_linter`, `uses_styling`, `deployed_on`, `owned_by`, `current_status`, `default_model`, `primary_branch`, `api_version`, `port_is`, `located_at`, `runtime_version` (plus built-in synonym aliases such as `auth_provider -> uses_auth`, `hosted_on -> deployed_on`, `db_engine -> uses_database`).
  - **Exclusive predicate groups:** `{enabled_in, disabled_in}`, `{depends_on, removed_dependency_on}`, `{approved, deprecated, rejected}`, `{active, archived}`.
- **`general_memory`**
  - **Functional:** `lives_in`, `works_at`, `role` (`has_role`, `job_title`), `cfo` (`has_cfo`, `serves_as_cfo_of`), `ceo` (`has_ceo`, `serves_as_ceo_of`), `reports_to`, `prefers`, `current_stage`, `headquartered_in`.
  - **Exclusive predicate groups:** `{employs, formerly_employed, terminated}`, `{married_to, divorced_from}`, `{allied_with, rival_of}`, `{owns, sold}`.

Reverting a prior decision (`Clerk -> BetterAuth -> Clerk`) or flipping polarity (`negated=False -> negated=True`) creates a fresh belief epoch rather than mutating retired edges, keeping historical `as_believed_at` queries exact.

---

## Configuration & CLI Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `POSTGRES_URI` | *(embedded PG on py3.12)* | PostgreSQL DSN (also reads `DATABASE_URL` or `RAG_DB_URI`). |
| `RAG_REALM` | `repo_<git-root>` | Tenant/project namespace. Auto-derived from the git repository root when unset. |
| `RAG_SPACE` | `default` | Sub-grouping space within the realm (`default`, `production`, `sandbox`, …). |
| `MCP_MEMORY_PRESET` | `engineering` | Initial rule preset (`engineering` or `general_memory`). |
| `MCP_EMBEDDING_BACKEND` | `auto` | `fastembed` (local ONNX) or `openai` (OpenAI-compatible `/v1/embeddings`). |
| `MCP_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local FastEmbed model (`384` dims; also supports `bge-base-en-v1.5`, `bge-m3`, etc.). |
| `MCP_ALLOWED_ROOTS` | current working dir | Path-separated list of directories `ingest_documents` is permitted to read. |
| `MCP_PGDATA_DIR` | `~/.local/share/post-graph-rag-mcp/pgdata` | Data directory for Tier-1 embedded PostgreSQL (`[embedded]`). |
| `RAG_STATEMENT_CACHE_SIZE` | *(asyncpg default)* | Set to `0` behind PgBouncer / Neon `-pooler` / Supabase transaction pooler (port `6543`). |
| `OPENAI_API_KEY` | *(unset)* | Enables Mode B autonomous LLM extraction and optional `synthesize=True` answers. |
| `OPENAI_API_BASE` | `http://localhost:4000/v1` | OpenAI-compatible router endpoint (`LiteLLM`, `OpenRouter`, `vLLM`, `Ollama`). |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http`. |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8080` | Bind address and port when running `streamable-http`. |

### CLI Commands

```bash
# Pre-warm embedded PostgreSQL (Python 3.12) and FastEmbed ONNX weights, and print IDE snippets
post-graph-rag-mcp init

# End-to-end health check (PostgreSQL + pgvector, realm schema, watermark, embedding dimensions)
post-graph-rag-mcp doctor

# Run MCP server (stdio by default, or Streamable HTTP)
post-graph-rag-mcp
post-graph-rag-mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/crajah/post-graph-rag/blob/main/LICENSE).

<!-- mcp-name: io.github.crajah/post-graph-rag-mcp -->
