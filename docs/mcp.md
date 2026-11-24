---
layout: default
title: "Using post-graph-rag with Claude, Gemini CLI, Cursor & MCP"
description: "Install and use post-graph-rag-mcp in Claude Code, Claude Desktop, Gemini CLI, Cursor, Windsurf, VS Code, and Codex CLI — bi-temporal Graph RAG memory on PostgreSQL + pgvector."
---

# Using `post-graph-rag` with Claude, Gemini CLI, Cursor & MCP

### Persistent, bi-temporal Graph RAG memory for AI coding assistants — on PostgreSQL + `pgvector`

**[GitHub](https://github.com/crajah/post-graph-rag)** · **[MCP Package on PyPI](https://pypi.org/project/post-graph-rag-mcp/)** · **[Core Library Write-Up](index.html)** · Apache 2.0

---

Standard agent memory stores only accumulate. Tell your assistant in October that `AuthService` uses Clerk, migrate to BetterAuth in November, and cosine similarity keeps returning both facts side by side.

`post-graph-rag-mcp` wraps [`post-graph-rag`](https://github.com/crajah/post-graph-rag) as a Model Context Protocol (MCP) server (`stdio` and Streamable HTTP). When a new decision lands, **the earlier assertion is deterministically closed** while both **valid time** (`as_of`) and **belief time** (`as_believed_at`) remain queryable for audit.

- **Zero sidecar LLM API keys on the interactive path**: your IDE or CLI assistant passes structured triples and keywords directly in the tool call while local ONNX embeddings (`fastembed`, `BAAI/bge-small-en-v1.5`, 384d) run in-process.
- **Three deployment tiers**: embedded local PostgreSQL (`uvx`, zero setup), your own PostgreSQL / Docker / [Neon / Supabase](https://github.com/crajah/post-graph-rag/blob/main/examples/00_neon_supabase_quickstart.md), or a shared team Streamable HTTP server via Docker Compose.

---

## 1. Choose Your Storage Tier

| Tier | Best for | What you need | Command / Config |
| :--- | :--- | :--- | :--- |
| **Tier 1: Zero-Config Local** | Trying it in 20 seconds on one machine | `uv` (`uvx`) only — bundles PostgreSQL 16.2 + `pgvector` (`pgserver`) & local ONNX embeddings | `uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp` |
| **Tier 2: Existing PostgreSQL** | Local PG 16–18, Docker, Neon, or Supabase (Python 3.11–3.14) | `POSTGRES_URI` pointing at a DB with `CREATE EXTENSION vector;` | `uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp` |
| **Tier 3: Shared Team Server** | Whole engineering team sharing one project memory over HTTP | Docker Compose | `docker compose up -d --build` → `http://localhost:8080/mcp` |

> **Why `--python 3.12` on Tier 1?** `pgserver` publishes prebuilt embedded-PostgreSQL wheels up to CPython 3.12 (`cp39–cp312`). Passing `--python 3.12` lets `uvx` fetch the right interpreter automatically even if your system Python is 3.13 or 3.14. Tier 2 (external `POSTGRES_URI`) requires no Python pin.

---

## 2. Add to Your AI Client

### Claude Code

**Tier 1 — Zero-config local (embedded PostgreSQL + local embeddings):**

```bash
claude mcp add post-graph-rag -- uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp
```

**Tier 2 — Existing PostgreSQL, Docker, Neon, or Supabase:**

```bash
claude mcp add post-graph-rag \
  -e POSTGRES_URI="postgresql://localhost:5432/postgres" \
  -- uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp
```

**Tier 3 — Shared team server (Streamable HTTP):**

```bash
claude mcp add --transport http post-graph-rag http://localhost:8080/mcp
```

Verify inside Claude Code with `/mcp` — you should see `post-graph-rag` connected with 7 tools.

---

### Gemini CLI

**Tier 1 — Zero-config local:**

```bash
gemini mcp add post-graph-rag uvx -- --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp
```

**Tier 2 — Existing PostgreSQL, Docker, Neon, or Supabase:**

```bash
gemini mcp add -e POSTGRES_URI="postgresql://localhost:5432/postgres" \
  post-graph-rag uvx -- --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp
```

**Tier 3 — Shared team server (Streamable HTTP):**

```bash
gemini mcp add --transport http post-graph-rag http://localhost:8080/mcp
```

Or configure Gemini CLI manually in its `settings.json` (`mcpServers` block):

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

Verify inside Gemini CLI with `/mcp list`.

---

### Claude Desktop

Add `post-graph-rag` to `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

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

Restart Claude Desktop. To use an external PostgreSQL database instead of embedded Postgres, remove `"--python", "3.12"`, change the package extra to `"post-graph-rag-mcp[local-embed]"`, and add `"env": {"POSTGRES_URI": "postgresql://..."}`.

---

### Cursor & Windsurf

- **Cursor:** `.cursor/mcp.json` (project-scoped) or `~/.cursor/mcp.json` (global)
- **Windsurf:** `~/.codeium/windsurf/mcp_config.json`

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

For an external PostgreSQL / Neon / Supabase database (including transaction poolers like Neon `-pooler` or Supabase port `6543`):

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

---

### VS Code

Add `.vscode/mcp.json` to your workspace (note that VS Code uses `"servers"` and `"type": "stdio"`):

```json
{
  "servers": {
    "post-graph-rag": {
      "type": "stdio",
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

---

### Codex CLI

```bash
codex mcp add post-graph-rag -- uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp
```

---

### Verify Your Setup (`init` and `doctor`)

Before opening a session, you can pre-warm the embedded PostgreSQL database and local FastEmbed ONNX weights (~30 MB, cached offline afterwards) and run an end-to-end health check:

```bash
# Pre-warm embedded Postgres + ONNX weights and print ready-to-paste client snippets
uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp init

# Live check: Postgres + pgvector, realm schema, embedding dimension probe, watermark
uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp doctor
```

On Tier 2 (external Postgres), omit `--python 3.12` and pass `POSTGRES_URI`:

```bash
POSTGRES_URI="postgresql://localhost:5432/postgres" \
  uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp doctor --json
```

---

## 3. Using the MCP: Real Session Walkthrough

Once connected, your assistant has **7 tools**, **3 resources**, and **3 prompts**. You talk to it in plain English — the assistant chooses the right tool automatically.

### Step 1 — Record an architectural decision (`remember`)

> **You:** "Remember that `AuthService` uses `Clerk` for authentication and is deployed on `Fly.io`, effective 2026-01-15."

The assistant calls `remember` with the passage and structured triples:
- `(AuthService) --[uses_auth]--> (Clerk)` (`valid_from: "2026-01-15"`)
- `(AuthService) --[deployed_on]--> (Fly.io)` (`valid_from: "2026-01-15"`)

Because `uses_auth` and `deployed_on` are built-in **functional predicates** in the default `engineering` preset, they are single-valued per subject.

### Step 2 — Migrate and automatically close the old decision (`remember`)

> **You:** "Remember that we migrated `AuthService` from Clerk to `BetterAuth` on 2026-06-01."

The assistant calls `remember` with `(AuthService) --[uses_auth]--> (BetterAuth)` (`valid_from: "2026-06-01"`).
- `post-graph-rag` **deterministically closes** the active `(AuthService) --[uses_auth]--> (Clerk)` edge (`superseded_by` + `t_expired`).
- The tool response returns both `superseded_relations` and `active_relations_on_subjects`, so if a synonym predicate was used earlier, the assistant can explicitly close it via `supersedes_edge_ids` in zero extra LLM calls.
- If you later revert (`Clerk → BetterAuth → Clerk`), a fresh belief epoch is opened rather than mutating the retired edge, keeping historical audit queries exact.

### Step 3 — Ask for current truth (`recall`)

> **You:** "What authentication provider does `AuthService` use?"

The assistant calls `recall`. Hybrid RRF retrieval (graph traversal + relation embeddings + `tsvector` full-text search) excludes superseded edges automatically and returns `BetterAuth` — never `Clerk` and `BetterAuth` competing as co-equal current facts.

### Step 4 — Time-travel & audit queries (`recall_as_of`)

> **You:** "What auth provider was `AuthService` using in March 2026?"

The assistant calls `recall_as_of(query="AuthService auth provider", as_of="2026-03-15")`.

> **You:** "What did this memory store believe on 2026-05-01 before the migration was recorded?"

The assistant calls `recall_as_of(query="AuthService auth provider", as_believed_at="2026-05-01T00:00:00Z")`.

Both axes are first-class:
- **`as_of`** = *valid time* (what held in the world on that date)
- **`as_believed_at`** = *belief time* (what the database knew at that timestamp)

### Step 5 — Trace full lineage in both directions (`trace_history`)

> **You:** "Show me the full auth history of `AuthService` — and tell me everywhere `Clerk` was ever used."

The assistant calls `trace_history` with bidirectional walks (`direction="both"`), returning `current_relations` and `superseded_relations` in chronological order — answering both outbound (*"what did AuthService use?"*) and inbound (*"what services depended on Clerk?"*) questions.

### Step 6 — Sync an ADR or docs directory (`ingest_documents`)

> **You:** "Ingest all Markdown ADRs under `./docs/adr` in filename order."

The assistant calls `ingest_documents(path="./docs/adr", order_by="filename")`.
- Files are ingested in deterministic order (`filename`, `mtime`, or `git_first_commit`) so `0007-switch-to-betterauth.md` cleanly supersedes `0002-use-clerk.md`.
- Re-running skips unchanged files in `O(1)` via content hash.
- Reads are sandboxed to `MCP_ALLOWED_ROOTS` (defaults to your current working directory).

### Step 7 — Poll changes & inspect rules (`changes_since`, `configure_memory_rules`)

- `changes_since(since=watermark)` returns a database-clock `CorpusDelta` of newly added or superseded relations, entities, and documents — no client clock skew.
- `configure_memory_rules` lets you inspect or extend active `functional_predicates` and `exclusive_predicate_groups` (`engineering` or `general_memory` presets), or check graph coverage and `dark_entities`.

---

## 4. Complete MCP Surface Reference

### 7 Tools

| Tool | Purpose |
| :--- | :--- |
| `remember` | Persist a decision, convention, or fact as an auditable passage + graph triples. Supports `functional=True`, `supersedes_edge_ids`, and returns `superseded_relations` + `active_relations_on_subjects`. |
| `recall` | Hybrid current-truth retrieval fusing graph walks, `pgvector` HNSW (chunks & relations), and `tsvector` GIN search via RRF. Excludes superseded and negated assertions automatically. |
| `recall_as_of` | Bi-temporal audit query by valid time (`as_of`) and/or belief time (`as_believed_at`). |
| `trace_history` | Bidirectional (`direction="both"`) lineage query returning `current_relations` and `superseded_relations` chronologically. |
| `ingest_documents` | Idempotent Markdown/ADR directory sync (`order_by="filename" \| "mtime" \| "git_first_commit"`), sandboxed to `MCP_ALLOWED_ROOTS`. |
| `changes_since` | Database-clock watermark delta feed (`CorpusDelta`) of new and superseded facts. |
| `configure_memory_rules` | Inspect or extend `functional_predicates` and `exclusive_predicate_groups`, or inspect coverage, dark entities, and document stats. |

### 3 Resources

- `memory://status` — Active `realm`, `space`, embedding backend & dimension, document/entity counts, and database watermark.
- `memory://rules` — Active preset, `functional_predicates`, `exclusive_predicate_groups`, `predicate_aliases`, and `predicate_vocabulary`.
- `memory://communities` — Hierarchical Leiden community summary tree (`get_community_tree()`).

### 3 Prompts

- `remember-architecture-decision` — Structured workflow for recording an architecture decision or migration with `functional=True` predicates and verifying `active_relations_on_subjects`.
- `audit-belief-state` — Side-by-side comparison of current truth (`recall`), point-in-time belief (`recall_as_of`), and full lineage (`trace_history`).
- `sync-adr-folder` — Guides the assistant through ordered ADR ingestion (`0001-*.md` → `0099-*.md`).

---

## 5. Two Operating Modes (Zero-Key vs Autonomous Router)

| Capability | Mode A: Zero-Key Host-Agent Mode *(Default)* | Mode B: Autonomous Router Mode *(Set `OPENAI_API_KEY`)* |
| :--- | :--- | :--- |
| **Triple extraction on `remember`** | Host assistant passes `triples` + `entities` directly (`0` sidecar LLM calls) | Host triples if passed; otherwise `GraphExtractor` extracts & gleans via `RAG_EXTRACTION_MODEL` |
| **Embeddings** | Local ONNX via `fastembed` (`BAAI/bge-small-en-v1.5`, 384d, offline after first pull) | Local `fastembed` by default, or `/v1/embeddings` when `MCP_EMBEDDING_BACKEND=openai` |
| **Supersession** | Functional predicates, exclusive predicate groups, polarity flips, and host-assisted `supersedes_edge_ids` | All deterministic passes **plus** optional `RAG_CONTRADICTION_DETECTION=1` LLM checks |
| **Retrieval (`recall` / `recall_as_of`)** | `GraphRAG.query_data()` with host keywords or deterministic lexical fallback (`0` chat-LLM calls) | Structured `query_data()` by default, or full `GraphRAG.query()` synthesis when `synthesize=True` |
| **Directory sync (`ingest_documents`)** | Client MCP sampling if supported; otherwise chunk vectors + YAML frontmatter triples | Full multi-chunk `GraphRAG.index_text()` with `DocumentContext` coreference threading |

To enable Mode B with OpenAI, OpenRouter, LiteLLM, vLLM, or Ollama, pass `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE` + `RAG_MODEL`) in your MCP client's `env` block.

---

## 6. Project Isolation, Presets & Environment Variables

By default, `RAG_REALM` is automatically derived from the git repository root (`repo_<name>`), and `schema_per_realm=True` gives each repository its own PostgreSQL schema so projects never leak memories into one another.

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `POSTGRES_URI` | *(embedded PG on py3.12)* | PostgreSQL DSN (also reads `DATABASE_URL` or `RAG_DB_URI`). |
| `RAG_REALM` | `repo_<git-root>` | Tenant/project namespace (isolated PostgreSQL schema). |
| `RAG_SPACE` | `default` | Sub-grouping within a realm (`default`, `production`, `sandbox`, …). |
| `MCP_MEMORY_PRESET` | `engineering` | Initial rule preset (`engineering` or `general_memory`). |
| `MCP_EMBEDDING_BACKEND` | `auto` | `fastembed` (local ONNX) or `openai` (OpenAI-compatible `/v1/embeddings`). |
| `MCP_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local FastEmbed model (`384` dims; also supports `bge-base-en-v1.5`, `bge-m3`, etc.). |
| `MCP_ALLOWED_ROOTS` | current working dir | Path-separated list of directories `ingest_documents` may read. |
| `MCP_PGDATA_DIR` | `~/.local/share/post-graph-rag-mcp/pgdata` | Data directory for Tier-1 embedded PostgreSQL (`[embedded]`). |
| `RAG_STATEMENT_CACHE_SIZE` | *(asyncpg default)* | Set to `0` behind PgBouncer / Neon `-pooler` / Supabase transaction pooler (port `6543`). |
| `OPENAI_API_KEY` | *(unset)* | Enables Mode B autonomous LLM extraction and optional `synthesize=True` answers. |
| `OPENAI_API_BASE` | `http://localhost:4000/v1` | OpenAI-compatible router endpoint (`LiteLLM`, `OpenRouter`, `vLLM`, `Ollama`). |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http`. |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8080` | Bind address and port for `streamable-http`. |

---

## 7. Shared Team Server (Docker Compose)

Run one shared bi-temporal memory server for your team (`pgvector/pgvector:pg17` + non-root `python:3.13-slim` Streamable HTTP container on port `8080`):

```bash
git clone https://github.com/crajah/post-graph-rag.git
cd post-graph-rag/post-graph-rag-mcp
docker compose up -d --build
```

Then point any client at `http://localhost:8080/mcp`:
- **Claude Code:** `claude mcp add --transport http post-graph-rag http://localhost:8080/mcp`
- **Gemini CLI:** `gemini mcp add --transport http post-graph-rag http://localhost:8080/mcp`
- **VS Code (`.vscode/mcp.json`):** `{"servers": {"post-graph-rag": {"type": "http", "url": "http://localhost:8080/mcp"}}}`

---

## 8. Troubleshooting

- **`pgserver` wheel build error on Python 3.13 / 3.14:** Add `--python 3.12` to your `uvx` command on Tier 1, or switch to Tier 2 (`POSTGRES_URI=...` with `post-graph-rag-mcp[local-embed]`, which supports Python 3.11–3.14 with no pin).
- **`DuplicatePreparedStatementError` on Neon or Supabase:** Set `RAG_STATEMENT_CACHE_SIZE=0` in your MCP `env` block when connecting through Neon's `-pooler` endpoint or Supabase's transaction pooler (port `6543`).
- **`SchemaError` mentioning embedding model mismatch:** Each realm records its `(embedding_model, embedding_dim)` fingerprint in `realm_metadata` so incompatible vector spaces are never mixed silently. Either use the original embedding model or pick a fresh `RAG_REALM`.
- **`ingest_documents` refused a path:** Set `MCP_ALLOWED_ROOTS` (colon-separated on macOS/Linux, semicolon-separated on Windows) to include the directory you want to sync.

---

**[Back to `post-graph-rag` Write-Up](index.html)** · **[`post-graph-rag-mcp` Source & README](https://github.com/crajah/post-graph-rag/tree/main/post-graph-rag-mcp)** · **[Neon & Supabase Quickstart](https://github.com/crajah/post-graph-rag/blob/main/examples/00_neon_supabase_quickstart.md)**
