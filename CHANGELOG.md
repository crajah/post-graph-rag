# Changelog

Releases before 1.13.0 are recorded in the git history and in the GitHub
releases page; this file starts where the first entry was written.

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
