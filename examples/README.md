# Examples

Seven runnable scripts, each demonstrating one capability. They share
configuration through `_shared.py`, so point them at your router and database
once:

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=https://your-router/v1        # any OpenAI-compatible endpoint
export POSTGRES_URI=postgresql://localhost:5432/postgres

pip install post-graph-rag
cd examples && python 01_quickstart.py
```

Each example uses its own realm, so they never collide and you can run them in
any order. Postgres needs the `vector` extension (`CREATE EXTENSION vector`).

| | what it shows | why it matters |
| :--- | :--- | :--- |
| [`01_quickstart.py`](01_quickstart.py) | Three documents, one question needing all three | The answer requires joining facts across documents — the graph walks relations a top-k chunk search would never assemble |
| [`02_supersession.py`](02_supersession.py) | Three filings, three CFOs, one current answer | A later document **closes** an earlier fact. History is kept and queryable, not overwritten |
| [`03_bitemporal_audit.py`](03_bitemporal_audit.py) | `as_believed_at` reproduces a past answer | "What did our systems report in March, before the restatement?" — the question auditors actually ask |
| [`04_multi_tenant_spaces.py`](04_multi_tenant_spaces.py) | Per-tenant isolation, plus a deliberate cross-tenant view | One database, many customers, no accidental leakage |
| [`05_incremental_and_deltas.py`](05_incremental_and_deltas.py) | Idempotent re-indexing and `changes_since()` | Re-indexing unchanged content is a no-op; polling tells you exactly what moved |
| [`06_communities_and_exploration.py`](06_communities_and_exploration.py) | Community reports, topic tree, coverage | Corpus-level questions, and finding where retrieval has never looked |
| [`07_retrieval_modes.py`](07_retrieval_modes.py) | `local`, `global` and `mix` on the same question | Entity-centred, corpus-level and fused retrieval answer differently — pick per question |

## Cost

These examples make real LLM calls. Extraction is the expensive part —
roughly one call per chunk indexed. The scripts here index between one and six
short documents each, so a full pass over all seven costs cents, not dollars,
on a small model.
