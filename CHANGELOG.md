# Changelog

Releases before 1.13.0 are recorded in the git history and in the GitHub
releases page; this file starts where the first entry was written.

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
