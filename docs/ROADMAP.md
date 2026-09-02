# Roadmap: exploration-support features

Three features, motivated by reading Nomad (arXiv:2603.29353): exploration-first
systems need a corpus substrate that exposes structure, coverage, and change.
post-graph-rag stays an engine — no agent loops enter the library — but each
feature below turns something an exploration consumer would otherwise build
badly on top of us into one engine call.

**Status: all three shipped in 1.10.0** (tasks #132–#134), with the noted
deviations closed properly: the level predicate is expression-indexed on all
realms, level filtering runs inside the vector search (post-graph 1.4.0
predicate pushdown), and the telemetry table lazy-creates on first write.
Still open from §3: the paired evaluation gate before `community_levels`
changes its default.

Build order was §1 → §2 → §3 as planned.

---

## 1. Corpus-delta API — `changes_since(T)` (task #132)

**Goal.** Answer "what changed since T" from belief time, so a consumer can
poll cheaply and re-explore only what moved. Nomad-style systems rebuild or
re-scan; nobody else carries `t_created`/`t_expired` on every relation, so
nobody else can answer this precisely.

**API.**
```python
delta = await rag.changes_since(
    "2026-09-01T00:00:00+00:00",
    space="production",              # optional; RESERVED_SPACE_ALL to widen
    include=("relations", "entities", "documents", "communities"),
    limit=500, after=None,           # keyset pagination on t_created (text-ordered ISO)
)
# CorpusDelta:
#   .new_relations          t_created > T, not superseded at read time
#   .superseded_relations   t_expired > T, each with .superseded_by id
#   .new_entities / .dormant_entities / .revived_entities
#   .new_documents          created_at > T (chunk-level, doc_key grouped)
#   .communities_stale      bool: oldest report build predates newest graph write
#   .as_of                  server watermark to pass as T next call
#   .counts                 dict, populated even in summary mode

n = await rag.changes_since(T, summary=True)   # counts only, no row transfer
```

**Design.**
- All predicates via post-graph 1.2.0 `find_vertices/find_edges ... where=`
  triples over payload keys: `("t_created", ">", T)`, `("t_expired", ">", T)`,
  `("superseded_by", "not_null", None)`, dormancy stamps. ISO-8601 UTC strings
  compare correctly as text — the 1.2.0 cast rule (str ⇒ text ordering) was
  built for exactly this shape.
- Summary mode uses `count_vertices` — no rows cross the wire; a poller runs
  it per tick and fetches detail only on nonzero.
- Watermark: `as_of` comes from the database clock in the same query round
  trip, not the client clock, so a poller never misses a write that committed
  between its clock and ours.
- At realm init (graph_store schema setup), call
  `create_payload_index(relations, key="t_created")` and `key="t_expired"`
  (text, not numeric) so delta polls are index scans. Idempotent by design.
- Relations predating the t_created field (older realms) are "always known"
  and never appear in deltas — documented, matching the belief-time default.
- Communities are rebuilt wholesale, so they get a staleness *flag* (we
  already compute this warning at retrieval) rather than a fake diff.

**Files.** `post_graph_rag/deltas.py` (new, ~150 lines), `models.py`
(CorpusDelta dataclass), `graph_store.py` (index creation at init, two
timestamp helpers), `engine.py` (thin `changes_since` delegating to deltas).

**Tests** (`tests/test_deltas.py`): new relation appears once and only once
across watermarked polls; supersession lands in `superseded_relations` with the
superseding id; dormancy and revival transitions; **re-indexing an unchanged
document yields an empty delta** (turns the idempotence claim into an
assertion); space scoping; pagination across a 3-page delta; summary counts
equal detail lengths; pre-field relations excluded.

**Risks.** Clock skew (mitigated by DB-clock watermark); realms created before
promoted temporal columns fall back to payload scans (correct, slower — note in
docs). Effort: ~1 day including tests.

---

## 2. Coverage / least-explored queries (task #133)

**Goal.** Expose which parts of the graph retrieval actually touches, so a
breadth-first consumer can pick the least-explored region — Nomad's topic
selection — with one call instead of client-side bookkeeping.

**API.**
```python
cov = await rag.coverage(space="production")
# [CommunityCoverage(community_id, title, members, retrieval_hits,
#                    last_hit_at, hit_share)]
frontier = await rag.least_explored_communities(space="production", k=5)
dark = await rag.dark_entities(space="production", limit=100)   # never retrieved
await rag.purge_retrieval_events(before="2026-08-01")           # retention
```

**Design.**
- **Opt-in telemetry**, `record_retrieval_events: bool = False`. When on,
  each query writes one event vertex to a `retrieval_events` table after the
  response is assembled: `{ts, mode, space, entity_ids, community_ids,
  query_sha256}` — the hash, never the query text, so no prompt content is
  persisted.
- **Best-effort by declared exception.** This library is fail-loud everywhere;
  telemetry is the one documented exemption: a failed event write logs a
  warning and never fails the query. The docstring says so and says why.
- Aggregation is SQL over the events table joined to `community_members`
  (one custom aggregate in graph_store; everything else via post-graph
  `count_vertices`/`find_vertices where=`). `hit_share` normalises by member
  count so big communities don't look explored merely by being big.
- `dark_entities`: entities with zero events and `dormant IS NULL` — the
  corpus nobody has ever asked about, which for an exploration consumer is
  the frontier.
- Retention: `purge_retrieval_events` is a thin wrapper over post-graph
  1.2.0 `delete_vertices(..., where=[("ts", "<", cutoff)])` — dogfooding the
  empty-where refusal and the range delete.

**Files.** `graph_store.py` (events table in schema init — created only when
the flag is on, so realms of non-telemetry users gain nothing), `engine.py`
(event emission at the end of `query_data`; coverage APIs), `config.py`
(flag), `models.py` (CommunityCoverage).

**Tests** (`tests/test_coverage.py`): flag off ⇒ zero writes (assert table
absent); one event per query across modes; hits aggregate per community;
least-explored ordering with a tie broken by size; dark entities shrink after
a targeted query; purge removes only pre-cutoff rows; a poisoned event write
(monkeypatched failure) does not fail the query but does log; space isolation.

**Risks.** Write amplification on hot query paths (one small insert per query,
opt-in, and documented); privacy (hash-only, stated). Effort: ~1–1.5 days.

---

## 3. Hierarchical communities — multi-level topic tree (task #134)

**Goal.** Replace the single-level community layer (a stated limitation since
v1.0) with a nested topic tree: fine clusters at L0, coarser themes above,
GraphRAG-style. Serves two masters: better global retrieval (drill-down instead
of flat ranking) and an exploration control state (Nomad's Topic Tree is this).

**Design.**
- **Recursive supergraph Leiden, not a resolution ladder.** Run Leiden at the
  current γ to get L0. Build a supergraph: one node per L0 cluster, edge
  weights = summed inter-cluster corroboration weights. Re-run Leiden on the
  supergraph → L1. Repeat to `community_levels` or until one cluster remains.
  Recursion *guarantees* nesting; independent runs at different resolutions do
  not, and a non-nested "hierarchy" poisons every drill-down.
- **Determinism holds**: Leiden seeded as today; supergraph construction is
  order-independent (sorted cluster ids).
- **Schema.** Communities keep their table; payload gains `level` (int) and
  `parent_id`. `level` is promoted via post-graph `promoted_keys=["level"]`
  so level filters are indexed. New edge table `community_children`
  (communities → communities), provisioned in graph_store alongside
  `community_members` (which stays L0-only: entities belong to leaves;
  ancestry is derived).
- **Reports.** L0 reports unchanged. L>0 reports are summarised **from child
  reports, not raw relations** — hierarchical summarisation caps cost growth
  (total extra LLM calls ≈ cluster count at L1+, typically ~15–30% more) and
  reads better: a theme report should synthesise findings, not re-derive them.
  Importance = max(child importances); a parent is as important as its most
  important child, never less. Partial failure keeps today's rule: skip the
  unusable child, build from the rest, largest-first.
- **Rebuild-not-accumulate extends to all levels**: one build clears every
  level for the space. The staleness warning is unchanged.
- **Retrieval.** `QueryParam.community_level: Optional[int]` — None keeps
  today's behaviour exactly (all levels retrieved, existing
  similarity/importance/size blend ranks them; size normalised within level so
  L2 giants don't smother L0 detail). Drill-down helpers:
  `get_community_tree(space)` (nested dict), `children_of(community_id)`.
  Level filtering compiles to a post-graph `where=[("level", "=", k)]`.
- **Config.** `community_levels: int = 1` — **default is current behaviour**,
  so this ships non-breaking and the hierarchy is opt-in until evaluated.

**Evaluation before any default change** — the paired discipline applies:
- "Main themes" answer quality, L1-report retrieval vs flat, judged blind on
  the Wikipedia + Dumas corpora.
- Largest-community share per level (the Leiden-vs-label-propagation table
  gains a level dimension).
- Report cost delta measured and published, not estimated.

**Files.** `communities.py` (recursion + child-report summarisation — the
bulk), `graph_store.py` (community_children, promoted level),
`engine.py`/`models.py` (QueryParam.community_level, tree APIs),
`config.py`, `reporting.py` (level-aware rendering).

**Tests** (`tests/test_hierarchy.py`): nesting invariant (every L_k community
has exactly one L_k+1 parent below the top); rebuild clears all levels;
determinism across two builds of the same graph; level filter returns only
that level; None-level equals pre-change results byte-for-byte on a fixture
realm (the non-breaking guarantee as an assertion); parent report cites only
child findings; partial child failure; `community_levels=1` produces zero new
tables rows beyond today's.

**Risks.** Cost growth (bounded by child-report summarisation, measured in
eval); hierarchy churn across re-index (documented: derived data, rebuilt);
global-mode regression (guarded by the paired eval gate before defaults move).
Effort: ~3–4 days including eval.

---

## Explicitly out of scope

Explorer/verifier agents, hypothesis generation, report stacks: application
layer. The engine exposes structure (§3), coverage (§2), and change (§1);
what explores them is the consumer's loop. At most, an `examples/` recipe
showing exploration state stored *as* a post-graph realm — typed hypothesis
vertices with audit history and supersession — once the three above exist.
