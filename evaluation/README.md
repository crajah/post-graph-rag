# Evaluation harness

Reproducible check that multi-document indexing and cross-document relationships
work, and a way to inspect extraction quality on real long documents rather than
toy paragraphs.

Unlike `tests/`, this needs a live LLM endpoint and costs tokens, so it is not
part of the test suite.

## Run

```bash
export POSTGRES_URI="postgresql://user@localhost:5432/postgres"
export OPENAI_API_BASE="http://localhost:4000/v1"
export OPENAI_API_KEY="..."

python evaluation/fetch_corpus.py
python evaluation/index_corpus.py --max-chunks 10 --model MiniMax-M2.7
python evaluation/analyse_graph.py
```

The corpus is downloaded rather than committed. The default four articles
(Ada Lovelace, Charles Babbage, Analytical engine, Difference engine) share many
entities, which is what makes them a test of cross-document resolution rather
than of extraction alone. Swap in your own:

```bash
python evaluation/fetch_corpus.py --articles Marie_Curie Pierre_Curie Radium
python evaluation/analyse_graph.py --ask "What did the Curies discover together?"
```

Indexing writes to its own realm schema (`--realm`, default `wiki_kb`) and
recreates it each run unless `--no-reset` is passed. Drop it with:

```sql
DROP SCHEMA IF EXISTS wiki_kb CASCADE;
```

Providers that exhaust credits mid-run are the normal case on shared routers.
Prefer **one model plus retries** — routers usually recover within seconds, and
`--max-retries` rides that out without silently degrading extraction quality.
`--fallback-models` exists for a genuinely prolonged outage where finishing
matters more than quality; list them in descending quality order.

`RAG_RETRY_DEADLINE` caps the wall-clock a single call may spend retrying. Without
it a sustained outage costs retries x models x backoff on *every* call — a
12-community build once spent 38 minutes producing nothing.

### Predicate vocabulary

Left unconstrained, extraction produces a predicate per relation — expressive, but
impossible to query by relation type. Constrain it with a bundled preset:

```bash
python evaluation/index_corpus.py --vocabulary-preset biography
```

or supply your own vocabulary and synonym map:

```bash
python evaluation/index_corpus.py \
    --predicate-vocabulary created designed worked_with influenced \
    --predicate-aliases '{"collaborated_with": "worked_with", "developed": "created"}'
```

The vocabulary steers the model and snaps morphological variants (`design` →
`designed`); the alias map merges genuinely different wordings for the same
relation. Inverse relations get their own canonical predicate — mapping
`educated_by` onto `taught` would silently reverse its direction.

## Community summarisation

Corpus-level questions are answered from summaries of clustered subgraphs, not
from retrieved passages. Run after indexing:

```bash
python evaluation/build_communities.py --realm wiki_kb --resolution 2.0 --synthesise
python evaluation/build_communities.py --ask "What themes connect these documents?"
```

On the reference corpus (428 entities, 449 relations) this detects 28 communities
and summarises the 12 largest.

**Partition balance depends heavily on the detector.** The largest community holds
35% of the graph under label propagation, 21% under Leiden at resolution 1.0, and
17% at resolution 2.0. A community covering a third of the graph is not a theme,
it is a summary of everything — which is why Leiden ships as a core dependency
rather than an opt-in. The label-propagation fallback remains for environments
where the native `igraph` build is unavailable.

**Report quality depends heavily on the model.** Same graph, same clustering, two
models:

| | Llama-3.3-70B | DeepSeek-V3.2 |
| :--- | :--- | :--- |
| Time | 20s | 56s |
| Distinct `rating` values | 1 (every community 8.0) | 4, spanning 6.0–9.0 |
| Duplicate titles | yes (2 identical) | none |
| Findings | generic ("Babbage's Work on Computers") | specific ("Allan G. Bromley's scholarship directly enabled physical reconstruction") |

The `rating` field is only meaningful with a model that discriminates. Check it
varies on yours before ranking by it.

## Comparing runs

Index the same corpus into different realms, changing one variable, then diff the
resulting graphs:

```bash
python evaluation/index_corpus.py --realm run_a --model MiniMax-M2.7
python evaluation/index_corpus.py --realm run_b --model gemma-4-31B-it
python evaluation/compare_realms.py run_a run_b
```

## Comparison against LightRAG

`compare_lightrag.py` indexes the same corpus with LightRAG so the two libraries
can be compared directly. LightRAG is not a dependency — install it separately.

Both libraries indexed the same four articles with the same model
(MiniMax-M2.7), the same embedding model (`text-embedding-3-small`, 1536 dims),
one gleaning pass, and equivalent chunk sizing (~2000 chars with ~200 overlap).
post-graph-rag ran with `max_concurrent_chunks=6`.

| | post-graph-rag (concurrent) | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| **Throughput** | | |
| Characters indexed | 66,345 | 75,751 |
| Indexing time | 3.7 min | **2.0 min** |
| Characters per minute | 17,900 | **37,900** |
| **Graph density** | | |
| Entities | **523** | 450 |
| Relations | **671** | 421 |
| Entities per 10k chars | **78.8** | 59.4 |
| Relations per 10k chars | **101.1** | 55.6 |
| Orphan entities | 24 | not exposed |
| **Edge labelling** | | |
| Distinct edge labels | **380** | 460 |
| Labels ÷ relations | **57%** | 109% |
| With a controlled vocabulary | **11%** | not supported |
| **Retrieval** | | |
| Query latency, mix / global | **8.2s / 3.1s** | 10.1s / 6.1s |

### Throughput

LightRAG is about **2.1x faster** per character. That is down from 6.4x: before
concurrency post-graph-rag processed chunks strictly sequentially and took
12.7 minutes on this corpus.

The residual gap is architectural rather than incidental. post-graph-rag
parallelises the network-bound work *within* a document and then applies graph
writes in order, because entity resolution is a read-modify-write against a
uniqueness index and concurrent writers would split entities that should merge.
LightRAG parallelises across documents as well, and its graph store is an
in-process NetworkX object rather than a transactional database, so it has no
equivalent constraint to respect.

That is a fair trade rather than a defect: the write serialisation is what makes
`Babbage` and `Charles Babbage` reliably converge on one vertex under
concurrency. Raising `max_concurrent_chunks` beyond 6 recovers some of the
difference, at the cost of coarser coreference context — chunks in the same batch
cannot see each other's entities.

### Graph density

post-graph-rag extracts roughly **33% more entities and 82% more relations per
unit of text**. The gap is widest on relations, which is where gleaning pays off:
a second pass asking what was missed materially raises relation recall.

Note the character counts differ — LightRAG indexed 14% more text, because
post-graph-rag's chunker discards section headings and short fragments. The
per-10k-character rows are the comparable ones.

### Edge labelling

This is the sharpest structural difference, and the one most likely to decide the
choice.

LightRAG's edge `keywords` are free text: 460 distinct labels across 421
relations, so *more than one label per edge on average*, with entries such as
"social contact" and "claimed influence". They describe an edge in prose but
cannot be queried as a relation type — `WHERE relation_type = 'worked_with'` has
no meaning against them.

post-graph-rag normalises predicates (case, separators, tense auxiliaries) and
optionally snaps them onto a controlled vocabulary. Unconstrained it sits at 57%;
with the `biography` preset it drops to **11%** — 44 labels over 417 relations,
with a real head to the distribution (`located_in` 38, `worked_with` 36,
`studied` 34). If you intend to traverse or filter by relation type, that
difference is decisive. If you only ever retrieve by similarity, it matters much
less.

### Retrieval

Answer quality is comparable. Both produce well-structured, accurate answers to
factual and comparative questions; post-graph-rag is faster at query time and
adds inline citations.

The libraries differ in how corpus-level questions are answered.
post-graph-rag's `global` mode ranks community reports by a blend of similarity,
importance and size. An earlier version ranked on similarity alone and drifted to
a niche cluster (steampunk alternate histories) on "what are the main themes?",
where LightRAG stayed on the central theme — a narrow cluster's summary can sit
closer to a short query than a broad one's precisely because it covers less
ground.

### Operational differences

**Storage.** post-graph-rag persists to PostgreSQL through post-graph, with
realm/space multi-tenancy, audit tables and append-only history. LightRAG
defaults to file-backed NetworkX plus nano-vectordb, with pluggable backends
(PostgreSQL, Neo4j, Milvus, Qdrant) available.

**Resilience.** On a router that exhausts provider credits mid-run, LightRAG's
first attempt failed 3 of 4 documents; the comparison only completed after adding
retry and failover to the harness. post-graph-rag has that built in, bounded by
`retry_deadline_secs` so a sustained outage surfaces quickly instead of burning
retries on every call.

**Failure semantics.** post-graph-rag skips an individual bad chunk but raises if
every chunk fails, so a total outage cannot be mistaken for successfully indexing
an empty corpus.

### Summary

Choose **LightRAG** for raw indexing throughput, or when a file-backed store is
enough and you retrieve purely by similarity.

Choose **post-graph-rag** for a denser graph, queryable relation types, and
transactional PostgreSQL storage with tenant isolation and audit history — at
roughly half the indexing throughput.

## What the analysis reports

1. **Corpus** — chunks, entities, relations, mentions.
2. **Cross-document entity resolution** — entities reached from more than one
   source document, via `doc_mentions`.
3. **Cross-document traversal** — outgoing relations on the most-shared entities.
4. **Predicate vocabulary** — distinct predicates and how many are used exactly
   once. A high share of single-use predicates means the graph is hard to query
   by relation type.
5. **Entity type vocabulary** — type sprawl and near-duplicate types.
6. **Likely unmerged aliases** — name groups that probably denote one real-world
   entity split across several vertices.
7. **Cross-document retrieval** — whether a question spanning two documents pulls
   chunks from both.

## Reference runs

Four articles, ~127k characters, 10 chunks each at 2000 chars,
`text-embedding-3-small`.

### Effect of each feature

Llama-3.3-70B throughout, so the columns isolate the features rather than the model:

| | baseline | + gleaning & aliases | + vocabulary |
| :--- | ---: | ---: | ---: |
| Chunks indexed | 40/40 | 39/40 | **40/40** |
| Entities | 403 | 447 | 428 |
| Relations | 321 | 450 | 449 |
| Distinct predicates | 227 | 341 | **240** |
| Relations on a vocabulary predicate | — | 12% | **44%** |
| Orphan entities (no relations) | 92 | 85 | **69** |
| Duplicate edge rows | 7 | 0 | 0 |
| Fragmented short-form vertices | 6 | 0 | 0 |
| Pronoun entities | 2 | 0 | 0 |

Reproduce the third column with `--gleaning-passes 1 --vocabulary-preset biography`.

### Effect of the model

Identical settings (gleaning 1, `biography` vocabulary, Leiden at resolution 2.0),
identical corpus, 40/40 chunks each — only the primary model differs:

| | Llama-3.3-70B | DeepSeek-V3.2 | MiniMax-M2.7 | gemma-4-31B |
| :--- | ---: | ---: | ---: | ---: |
| Indexing time | **4.8 min** | 12.5 min | 12.7 min | 11.9 min |
| Entities | 428 | 447 | **488** | 299 |
| Relations | 449 | 599 | **705** | 417 |
| Entities carrying aliases | 47 | 139 | **268** | 120 |
| Negated relations captured | 0 | **21** | 7 | 14 |
| Relations corroborated (weight > 1) | 16 | 35 | 28 | **44** |
| Orphan entities | 69 | 41 | **26** | 31 |
| Distinct predicates | 240 | 279 | 395 | **44** |
| Single-use predicates | 74% | 67% | 74% | **25%** |
| Relations on vocabulary | 44% | 41% | 33% | **94%** |
| Community `rating` spread | 1 value | 3 (7–9) | 3 (7–9) | **6 (4–10)** |

There is no single winner — the models split into two useful shapes.

**MiniMax-M2.7 builds the richest graph**: most entities, most relations, and 268
entities carrying aliases (5.7x Llama), which is what drives its low orphan count.
Its community reports are the most insightful — findings like "ENIAC designers
worked independently without knowledge of Babbage's analytical engine" rather than
restatements of the cluster label.

**gemma-4-31B builds the most queryable graph**: 44 distinct predicates at 94%
vocabulary adherence, against 33–44% for everything else. Its top predicates carry
real mass — `located_in`(38), `worked_with`(36), `studied`(34) — where MiniMax's
heaviest predicate appears only 19 times. It also produced the most corroborated
edges (44) and by far the best community `rating` discrimination (6 distinct values
spanning 4–10). The cost is recall: it stores 40% fewer relations than MiniMax,
because it snaps aggressively onto the vocabulary rather than preserving
distinctions.

Pick on what you need: **MiniMax if you will query by similarity and traversal,
gemma if you will query by relation type.** Llama-3.3-70B is dominated on every
axis and is only worth using as a last-resort fallback.

A separate lesson: features that depend on the model doing something subtle fail
silently on weak models. Aliases only merge if the model emits them, and Llama
never once used the `negated` field on this corpus — silently collapsing "X worked
with Y" and "X never met Y" into the same edge.

Cross-document linkage works throughout: `Charles Babbage` and `Analytical Engine`
are each reached from all three articles that mention them, and a question about
the two engines retrieves chunks from both engine articles.

Gleaning raises relation recall about 40%. Alias resolution removes short-form
fragmentation entirely — `Charles Babbage` absorbs `Babbage` and `Mr. Babbage`,
`Ada Lovelace` absorbs six surface forms.

The vocabulary is what makes the graph queryable by relation type: without it the
head of the distribution is flat (the commonest predicate appears 6 times), with it
the head is real — `designed`(20), `worked_with`(18), `built`(17), `wrote`(14).
A long tail of single-use predicates survives by design; those are genuinely
distinct relations (`funded`, `lived_at`, `co_authored`) that a 35-term vocabulary
does not cover. Section 4 of the analysis lists the commonest off-vocabulary
predicates, which is the feedback loop for extending the preset.
