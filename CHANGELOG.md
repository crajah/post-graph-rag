# Changelog

Releases before 1.13.0 are recorded in the git history and in the GitHub
releases page; this file starts where the first entry was written.

## 1.20.2

### Changed

- Require `post-graph >= 1.16.0`, eliminating unconditional `to_jsonb(t)` TOAST
  payload decompression and full-row JSONB serialization on reads (`crajah/post-graph#5`)
  and enabling server-side JSONB key and vector projection (`crajah/post-graph#6`).
- Adopt `include_embedding=False` and `payload_fields` across `RAGGraphStore`,
  `GraphRAG`, and `CorpusDelta` read/search/traversal paths (`search_similar_entities`,
  `search_similar_documents`, `search_similar_communities`, `search_similar_relations`,
  `documents_stats`, `supersede_*`, `get_active_subject_relations`, `get_neighbors`,
  `get_relations_by_ids`, `community_children`, and delta feeds) so unused vector
  columns and unrequested chunk/entity blobs are omitted inside PostgreSQL before
  wire transfer.

## 1.20.1

### Changed

- Require `post-graph >= 1.15.0`, which replaces the unindexed `SELECT MAX(id)`
  sequence sync on explicit-ID `add_vertex` / `add_edge` writes with an `O(1)`
  monotonic `pg_sequence_last_value` advance (eliminating full-table scans and
  space-scoped RLS sequence rewinds; `crajah/post-graph#4`).

## 1.20.0

### Added

- **Functional predicates (`RAGConfig.functional_predicates`, `RAG_FUNCTIONAL_PREDICATES`)**:
  single-valued predicates per subject (e.g. `uses_auth`, `cfo_of`, `lives_in`).
  When a new positive assertion lands with a different object, active positive
  edges sharing `(subject, predicate)` are closed automatically via
  `RAGGraphStore.supersede_functional()`. Objects of functional predicates also
  bypass bare-quantity rejection so scalar values such as ports or versions
  survive extraction validation.
- **Polarity-flip supersession & multi-epoch belief histories**:
  `RAGGraphStore.supersede_opposite_polarity()` closes an active edge when the
  opposite polarity (`negated=False` vs `negated=True`) is asserted on the same
  triple. `add_relation()` now deduplicates only against *active* same-polarity
  edges, so reverting a prior decision (`A -> B -> A`) opens a new belief epoch
  rather than mutating the closed edge — keeping historical `as_believed_at`
  queries exact.
- **Caller-supplied extraction (`GraphRAG.index_document(..., extraction=...)`)**:
  accepts a pre-built `ExtractionResult` directly so host-agent or structured
  callers can index passages and triples without invoking an extraction LLM.
- **Explicit retraction and active-edge inspection**:
  `GraphRAG.retract_relations()`, `GraphRAG.active_relations_for_entities()`,
  `GraphRAG.set_functional_predicates()`, and `RAGGraphStore.get_active_subject_relations()`.
- **Bidirectional graph traversal (`QueryParam.direction`)**:
  supports `"out"` (default), `"in"`, and `"both"` across `get_neighbors()`,
  `get_neighborhood()`, and `query()` / `query_data()`, enabling reverse
  lookups (*"where was Clerk used?"*).
- **Embedding model fingerprinting (`RAGConfig.enforce_embedding_model_match`,
  `RAG_ENFORCE_EMBEDDING_MODEL`, default on)**: records `(embedding_model,
  embedding_dim)` in `realm_metadata` during `initialize_schema()` and raises
  `SchemaError` if a realm is reopened under a different embedding model of the
  same width.
- **`post-graph-rag-mcp 0.1.0`**: Model Context Protocol server (`stdio` and
  Streamable HTTP) in `post-graph-rag-mcp/` exposing 7 tools, 3 resources, and
  3 prompts with local ONNX embeddings (`fastembed`) and optional embedded
  PostgreSQL 16 + `pgvector`.

### Fixed

- **Graph-level `as_believed_at` and interval-bounded `as_of` recall**:
  `as_believed_at` now includes edges that were active at `T` even if closed
  later, and `as_of` respects valid-time intervals cleanly across retrieval
  channels.
- **`supersede_conflicting` idempotency guard**: skips already-closed edges
  (`superseded_by IS NULL`) so historical epochs are never overwritten.
- **Multi-epoch deduplication**: `_dedupe_triples` and `_merge_by_rrf` key by
  `edge_id` so distinct belief epochs of the same `(subject, predicate, object)`
  do not collapse during history queries.

## 1.19.2

### Fixed

- Corrected README metadata on PyPI, including the Graphiti comparison table
  entry and stale release details.

## 1.19.1

### Changed

- Updated CI matrix and version classifiers to test Python 3.11 and 3.13
  against PostgreSQL 16 and 18 with `pgvector`.

## 1.19.0

### Added

- `RAGGraphStore.delete_document_chunks(..., purge_audit=True)` removes the
  shadow `_audit` rows for deleted `documents` and `doc_mentions` after the
  live deletes complete, returning `audit_rows_purged` in the receipt.

## 1.18.0

### Added

- `RAGConfig.statement_cache_size` (`RAG_STATEMENT_CACHE_SIZE`) and
  `RAGConfig.server_settings` forwarded to `asyncpg.create_pool`, enabling
  safe operation behind PgBouncer in transaction or statement pooling mode.
- `RAGConfig.owner_role` (`RAG_OWNER_ROLE`) and
  `RAGGraphStore.reassign_owned_to(role)` so a subordinate batch writer that
  provisions a realm hands object ownership back to the owning role.

## 1.17.1

### Changed

- Packaging bump requiring `post-graph >= 1.11.0`.

## 1.17.0

### Fixed

- **No DDL on the query path**: `initialize_schema()` creates every index and
  optional telemetry table upfront so a non-owner role with forced row-level
  security can retrieve without hitting PostgreSQL's `CREATE INDEX` ownership
  check.
- **Duck-typed pool wrappers**: `GraphRAG(..., connection=...)` now accepts any
  pool wrapper exposing `acquire()` rather than requiring an `asyncpg.Pool`
  subclass.

## 1.16.0

### Added

- **Bring your own connection (`GraphRAG(..., connection=...)`)**: pass an
  existing `asyncpg.Connection`, `asyncpg.Pool`, or pool wrapper to share pools
  or assume per-tenant roles (`SET ROLE`) without the library taking ownership
  of connection lifecycle.

## 1.15.3

### Fixed

- ECT-QA period parsing now normalises months, half-years, and fiscal
  years/quarters consistently.
- Made answer truncation limit configurable in the evaluation harness and
  bumped minimum `post-graph` dependency to `1.10.1`.

## 1.15.2

### Fixed

**An embedding failure on a chunk's derived batches lost the passage.**
`_prepare_document` embeds the chunk, then embeds what the extractor derived
from it -- entities, stub endpoints, relations. A failure on that second set
aborted the chunk, discarding the chunk's own vector, which had already been
computed successfully. The vector channel never needed entities: the same
argument that keeps an unextractable passage now keeps an unembeddable one. The
passage is written with its own vector and no structure.

**The cause: embedding requests were never batched.** `get_embeddings` sent the
whole list in one request. Endpoints cap inputs per request and exceeding the
cap is a 400 -- not retryable -- so a chunk that extracted more entities than
the cap lost its entire passage on a limit that had nothing to do with the
passage. Requests are now split to `embedding_batch_size`
(`RAG_EMBEDDING_BATCH_SIZE`, default 64). Without this, the tolerance above
would merely have traded lost passages for quietly lost structure on every
dense chunk.

The chunk's own embedding failing still fails the chunk. A passage stored with
no vector is in the corpus and retrievable by nothing, which is the failure all
of this exists to prevent.

**A failure after the passage was written raised, leaving a partial write and
reporting the chunk as skipped.** Seven operations in `_write_document` run
after `add_document`, one of them an LLM call (contradiction detection inside
supersession). Any of them raising produced a state worse than losing the
chunk: the passage stored, structure partial, the chunk reported skipped -- so
a caller comparing posted against indexed counts fired on a mismatch that was
actively wrong about what was stored.

Structural writes now degrade at their own granularity and are counted. A
failing entity write costs that entity; a failing relation write costs that
triple; a failing supersession or contradiction check costs neither the
passage nor the relation it was enriching.

### Added

- `embedding_batch_size` (`RAG_EMBEDDING_BATCH_SIZE`, default 64).
- Results carry `embeddings_degraded: bool` -- structure was extracted but
  could not be embedded, so it was not written. Distinct from
  `extraction_empty`, which says no structure was written without saying why.
- Results carry `structure_errors: int` -- structural writes that failed after
  the passage was stored. Zero on a healthy run. **Aggregate it:** a corpus
  that indexed every chunk with `structure_errors` on every chunk is complete
  and useless, and no other count reports that.

The invariant behind all three fixes is now recorded at the top of the indexing
section of `engine.py`.

## 1.15.1

### Fixed

**A multi-replica deployment could not provision a new realm.** Every replica
calls `initialize_schema`, and `_ensure_entity_name_index` raised a
`SchemaError` for all but one of them: `IF NOT EXISTS` checks the catalog and
then creates, so more than one session passes the check and the losers collide
on `pg_class`. Measured against a live cluster at six replicas, five failed.
The relation-sources index logged a warning on the same race.

Both now tolerate a concurrent creator. Losing that race is not a failure --
the index exists afterwards, which is the whole contract -- while a genuine
duplicate-rows conflict still raises, because that one means entity resolution
cannot be enforced and must not be swallowed.

### Changed

Requires `post-graph >= 1.8.0`, which makes table, index and schema creation
tolerate a concurrent creator throughout and exports `is_concurrent_creation`
so this package can apply the same rule to its own provisioning.

Verified live at 6, 12, 20 and 32 replicas initializing one realm at once: all
succeed, and the realm is writable afterwards.

## 1.15.0

### Added

**Structure can come from your own code, not only from the model.**
`RAGConfig.extraction_fn` takes a callable `(text, context)` returning triples.
Imports, call edges, ownership and deployment topology are recoverable exactly
from an AST, an LSP index or a build file, and asking a model to infer them is
strictly worse than reading them. Extraction then earns its cost on the
unwritten part -- rationale, causation, what a decision was for.

The function may be sync or async, and may return an `ExtractionResult`, an
`{"entities": [...], "triples": [...]}` mapping, a bare sequence of triples, or
`None` for "nothing here". Endpoints not also described as entities become
stubs, the same path the model's own unmatched endpoints take, so a
triples-only source still produces a traversable graph.

`extraction_fn_mode` (`RAG_EXTRACTION_FN_MODE`, default `replace`) chooses
whether the function replaces the LLM or runs alongside it. Under `merge` both
run and their results are unioned, which is how a deterministic code graph and
the model's reading of the prose around it end up in one graph. Each source is
gated on its own way in, so an exact external record is not held to the LLM's
vocabulary and an LLM record is not let through ungated.

`validate_external_extraction` (`RAG_VALIDATE_EXTERNAL`, default on) puts
externally supplied records through the same gates as the model's: pronominal
and phrase-shaped names dropped, vague predicates and self-loops rejected,
predicates snapped onto the vocabulary. That is what makes "extraction output
is untrusted input" true of every source rather than only of the model. Turn it
off when the source is already exact and the vocabulary would mangle it --
`calls` and `imports` from an AST do not want snapping.

Two behaviours worth knowing. An exception from the function fails the chunk,
deliberately: a broken external extractor is the same event as a dead model,
and swallowing it would let an import that stopped resolving read as a corpus
with no call edges in it. Returning nothing, by contrast, keeps the passage,
consistent with 1.14.2. Relations from the function carry the chunk that
asserted them, so deleting that document withdraws them exactly as it would the
model's.

Nothing changes for a configuration that does not set `extraction_fn`.

## 1.14.2

### Fixed

**A chunk whose extraction yielded nothing lost its passage as well as its
structure.** `_prepare_document` gathered the embedding and the extraction
together, so an `ExtractionError` discarded the embedding too and the chunk was
skipped entirely. The text was then in the corpus and retrievable by nothing --
absent from every count, and indistinguishable from text that was never
ingested. Measured on a live corpus at 3 chunks in 180.

`ExtractionError` from the extractor is now treated as an empty extraction
rather than a failed chunk: the passage and its embedding are written, and it
stays retrievable by vector search. The refusal to invent structure is
unchanged -- an empty extraction writes no entities and no relations, and the
extractor still raises for callers using it directly.

**Only `ExtractionError`.** A timeout, a dropped connection and a malformed
response still fail the chunk exactly as before. Collapsing those into "the
passage had no entities" would let a dead extraction model read as a corpus
with no entities in it, silently, which is worse than the bug being fixed.
`_assert_any_progress` still refuses a run in which nothing was indexed.

`index_document`, the single-chunk method, behaves the same way rather than
raising. It is the same bug seen one chunk at a time, and a caller handed an
exception cannot recover the embedding already paid for, whereas a caller who
wants this treated as fatal can read the new field and raise.

### Added

Results carry `extraction_empty: bool`, so a caller can count these and tell
"nothing extractable here" apart from "never indexed".

## 1.14.1

### Fixed

**A relation retired by a deletion stayed retired when a later document
asserted it again.** `add_relation` rebuilt the payload without carrying or
clearing `dormant_since`, and the write merged rather than replaced, so a
relation whose last source had been deleted took the new chunk into `sources`,
had its weight go back up, and remained dormant -- invisible to every read
path while a live document said it held.

A documented re-assertion now revives it: dormancy is cleared and `revived_at`
stamped. Both dormancy reasons revive this way, since a document asserting the
relation outright is stronger evidence than the orphaned-endpoint rule that may
have retired it. A direct store write with no `source_chunk` does not revive --
that is not a corpus saying the relation holds again, and it must not undo a
deletion.

The payload is now built from the stored one and written with `replace=True`,
so keys written by other paths (a supersession pointer, transaction time)
survive while the two dormancy keys are genuinely removed. Merging cannot
remove a key, which is why the previous write could not express this.

### Changed

Requires `post-graph >= 1.7.0` for the `replace=True` write above, and for the
auditing controls it adds: a table can decline its shadow audit table
(`create_vertex_table(..., audited=False)`) and an existing one can be dropped
(`drop_audit_table`).

## 1.14.0

### Added

**Per-role model selection.** One model served every LLM call, which is a
compromise no role actually wants. `RAGConfig` now takes four optional
overrides, each falling back to `model` when unset:

- `extraction_model` (`RAG_EXTRACTION_MODEL`) — extraction, gleaning and
  contradiction detection, the index-time write path.
- `community_model` (`RAG_COMMUNITY_MODEL`) — community report synthesis.
- `auxiliary_model` (`RAG_AUXILIARY_MODEL`) — keyword extraction and query
  decomposition, the small per-query calls.
- `synthesis_model` (`RAG_SYNTHESIS_MODEL`) — answering the question.

The roles have different optima and different economics. Over one fixed graph
four answering models separate across a 23-point band, while on the extraction
side the models disagree about what a good graph is at all — one builds the
richest, another the most queryable at 40% fewer relations. Extraction runs one
or two calls per chunk and runs once; the query-time roles run on every question
for ever.

`RAGConfig.model_for(role)` resolves a role, and `LLMService.chat_completion`
and `chat_completion_stream` take an optional `role`. An unknown role raises
rather than falling back, so a typo at a call site cannot route that role to the
default model indefinitely while appearing to work.

Nothing else changes: no pipeline stages were added and no retrieval or
assembly behaviour was touched. A configuration with no overrides resolves every
role to `model` and behaves exactly as before, which is covered by a test that
indexes and queries end to end and asserts a single model was used throughout.

Beware `extraction_model` specifically: a weak model there does not fail, it
silently degrades the graph. One tested model never emitted the `negated` field,
collapsing "X worked with Y" and "X never met Y" into the same edge.

## 1.13.1

Requires `post-graph >= 1.6.0`, which adds `replace=True` to `upsert_vertex` and
`upsert_edge`. Without it a payload write could not remove a key: the write
merged into the stored payload, so writing a payload without a key left the old
key in the row. No behaviour in this package changes with this release.

## 1.13.0

### Fixed

**Relations survived document deletion.** Deleting a document marked its
relations dormant and then every read path returned them anyway, so to a
caller nothing had been deleted. `search_relations_text`, `get_all_relations`,
`get_relations_by_ids` and `get_neighbors` filtered neither dormancy nor
expiry, and `get_neighborhood` filtered expiry but not dormancy. All of them
now exclude dormant relations by default, as SQL predicates rather than
post-filtering, and take `include_dormant=True` for audit callers. Community
building excludes them too, without which a deleted document reappeared
through the summaries built over its relations.

**Relations written before provenance existed could never be retired.** Those
rows carry no `sources`, so withdrawal had nothing to match and no document
deletion could ever reach them — silently, and permanently. Provenance cannot
be reconstructed after the fact, so the graph is used instead: a relation whose
endpoint entities have *both* gone dormant is now marked dormant too, stamped
`dormant_reason: "orphaned_endpoints"` to distinguish it from source
withdrawal. `sweep_orphaned_relations(space)` applies the same rule across a
space, so an existing deployment repairs its stranded rows without
re-indexing. Revival is symmetric: a relation retired for that reason comes
back when either endpoint is mentioned again, while one whose sources were
withdrawn stays dormant until a document asserts it afresh.

**Withdrawal was O(all relations in the space) per document deletion.** It
loaded every relation and filtered in Python, so the cost of deleting one
document grew with the size of the whole graph. It is now a single set-based
statement over a `jsonb ?|` test against a new GIN index on
`payload->'sources'`, touching only rows whose provenance names a deleted
chunk. Source identifiers are coerced to text on both sides at write and at
match, so a future non-string chunk id cannot strand provenance the same way.

Nothing is physically deleted by any of this. Relations are marked dormant and
remain reachable with `include_dormant=True`, which is what the audit trail is
for.

### Added

- `GraphRAG.document_stats(doc_key, space=None) -> DocumentStats` — chunks,
  bytes, entities (split current/dormant), relations contributed, and first and
  last indexing times for one document. An absent document returns zeros with
  `found=False` rather than raising, so "indexed but empty" and "not present"
  stay distinguishable.
- `GraphRAG.documents_stats(doc_keys, space=None)` — the batch form, two SQL
  round trips for any number of keys rather than a query per document.
- `GraphRAG.document_graph(doc_key, space=None)` — the entities and relations
  one document contributed, for a drill-down view. Output is capped at 500
  entities and 1000 relations by default and reports `truncated`.
- `GraphRAG.sweep_orphaned_relations(space=None) -> int` — the one-shot repair
  described above.
- `DocumentStats` is exported from the package root.
- `initialize_schema` now creates a payload index on `relations.dormant_since`
  and a GIN index on `relations.payload->'sources'`, both idempotently, so
  existing realms gain them on next start.

Per-document reads refuse `RESERVED_SPACE_ALL`: they resolve a caller-supplied
document key, and doing that across every space would return another tenant's
document under the key this one asked for.
