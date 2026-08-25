---
layout: default
title: "post-graph-rag: Graph RAG with a Memory of Time"
description: "A Graph RAG library on PostgreSQL that beats Graphiti's published LongMemEval numbers at its model tier — with facts that expire, on one database you already run."
---

# post-graph-rag: Graph RAG with a Memory of Time

### Beats Graphiti's published LongMemEval scores at the same model tier. Runs entirely on PostgreSQL. Understands that facts expire.

**[GitHub](https://github.com/crajah/post-graph-rag)** · **[PyPI](https://pypi.org/project/post-graph-rag/)** · `pip install post-graph-rag` · Apache 2.0

---
---

Almost every Graph RAG system treats extracted relations as **timeless assertions**.

Consider a corpus where one document says two people are allies and a later document says they became rivals. Both edges land in the graph. Neither carries a period, an ordering, or any notion that the second contradicts the first. Retrieval hands the model both as co-equal current facts, and it reports — correctly, given what it was given — that they are allies *and* rivals.

This is not an edge case. It is the normal condition of any corpus that spans time: employment histories, org charts, contracts, case law, news archives, research literature. Facts expire. Graphs built from them usually cannot express that.

`post-graph-rag` closes the earlier assertion instead of storing both. Indexing a trilogy of novels in publication order, it closed **13 relationships** that later books contradicted — resolving from document order alone, without the extractor supplying a single date. Run over the same corpus, LightRAG produced 1,447 relations in which those same contradictions all coexist as equally current facts, because it has no mechanism to express that one supersedes another.

There is a second, more mundane problem. Ask an LLM for relations and you get beautiful, unusable variety. LightRAG over a four-article corpus produced 421 relations carrying **460 distinct edge keywords** — more than one unique label per edge, with entries like "social contact" and "claimed influence". An unconstrained run through post-graph-rag's own extractor gave 395 predicates across 705 relations, 74% used exactly once. Both graphs are readable. Neither is queryable: `WHERE relation_type = 'worked_with'` matches nothing useful when every edge carries a bespoke label.

## What post-graph-rag is

`post-graph-rag` is an open-source Graph RAG library that runs entirely on PostgreSQL. It indexes documents by extracting entities and relations with an LLM, stores them as a property graph alongside pgvector embeddings, and answers questions by combining vector similarity with graph traversal — plus LLM-generated summaries of clustered subgraphs for corpus-level questions.

The architectural bet is that **you do not need a separate vector store or graph engine**. pgvector provides HNSW similarity search; PostgreSQL provides transactions, JSONB and foreign keys; a property graph is two tables. One database, one consistency model, one backup.

What sits on top of that substrate is aimed at the two problems above — a temporal model where a later assertion can close an earlier one, and predicate vocabularies that make relation types queryable — plus community summarisation for corpus-level questions.

To be precise about what is novel here, since the field moves quickly:

| | GraphRAG | LightRAG 1.5.6 | post-graph-rag |
| :--- | :---: | :---: | :---: |
| Community detection + summaries | ✅ | ❌ | ✅ |
| Temporal model (validity, supersession) | ❌ | ❌ | ✅ |
| Controlled predicate vocabulary | ❌ | ❌ | ✅ |
| Single-datastore (graph + vectors + transactions) | ❌ | partial | ✅ |

Community summarisation is Microsoft GraphRAG's contribution, adopted here rather than invented; LightRAG dropped it in favour of dual-level keyword retrieval, so it is a genuine difference against that system but not against GraphRAG. The temporal model and the predicate vocabulary appear in neither.

This article walks through those mechanisms, along with entity resolution and concurrency. Every claim carries the measurement behind it: against LightRAG on identical corpus, model and embeddings, and against Zep's published LongMemEval numbers on the full 500-question set. Where a measurement was wrong, or an improvement failed, that is here too — the failures turned out to be the more useful half.

---

## The numbers

[LongMemEval](https://arxiv.org/abs/2410.10813) is the hardest public test of whether a system remembers correctly across long, changing conversations. It is also the benchmark Zep publish on for Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)) — so the comparison is against numbers their team chose to stand behind.

**Full 500-question set. All six question types. Nothing sampled.**

| | overall | multi-session | temporal | knowledge-update |
| :--- | ---: | ---: | ---: | ---: |
| **post-graph-rag** · `gemini-3.7-flash` | **68.3%** | **66.2%** | 52.6% | 70.5% |
| Zep/Graphiti · gpt-4o | 71.2% | 57.9% | 62.4% | 83.3% |
| Zep/Graphiti · gpt-4o-mini | 63.8% | 40.6% | 36.5% | 76.9% |
| Full-context baseline · gpt-4o | 60.2% | 44.3% | 45.1% | 78.2% |

`gemini-3.7-flash` is a small, cheap, fast model — the gpt-4o-mini tier. **Against that tier post-graph-rag wins five of six categories and takes overall by 4.5 points.** Against gpt-4o, a tier above, it lands within 2.9 points of Graphiti and **beats it outright on multi-session** — questions whose answer is scattered across separate conversations, the hardest category in the set and the one Graphiti scores lowest on.

That multi-session result is the one to look at twice: **66.2% against 57.9%**, and against the mini tier, **+25.6 points**. Cross-session synthesis is exactly what a temporal knowledge graph is supposed to buy you, and it is where the margin is widest.

Median query latency: **7.2 seconds**, on a laptop, against local PostgreSQL. No separate memory service, no graph engine to operate.

**Three qualifications, stated because you would find them anyway.** Zep judge with GPT-4o where this uses a three-model majority panel. Their generation models are a tier above `gemini-3.7-flash` — which is the point of the tier comparison, but it cuts both ways. And one question of 500 is excluded, a session neither extraction prompt could turn into triples; the harness refuses to call a run reportable while that count is nonzero.

Everything needed to reproduce this ships in the repo: the harness, the frozen configuration, the judge panel, and every failing case.

---
## What you can build with it

The temporal model is not a feature looking for a use case. These are the shapes it exists for:

**Agent memory that does not contradict itself.** An assistant that has talked to a user for six months has been told the same thing several ways, and the later telling usually wins. post-graph-rag closes the earlier assertion instead of retrieving both — which is why multi-session is its strongest category rather than its weakest.

**Anything with an as-of question.** Employment histories, org charts, contracts, policy versions, case law, pricing. "Who owned this in March?" and "what did we believe in March?" are different questions, and both are answerable because every relation carries validity time *and* belief time.

**Financial and regulatory corpora.** Sixteen quarters of earnings calls restate the same metrics with different values. Without a date on every fact, a figure is indistinguishable from fifteen others. With one, the quarter *is* the key.

**Anywhere the graph must live beside your data.** One PostgreSQL instance means a transaction can span your graph and your application tables — something no external graph engine can offer, at any price.

```python
from post_graph_rag import GraphRAG, RAGConfig

rag = GraphRAG(RAGConfig(realm="my_app", schema_per_realm=True))
await rag.initialize()

await rag.index_document(text, metadata=DocumentMetadata(
    source="/corpus/2024-q1.txt", document="2024-q1"))

answer = await rag.query("What changed since last quarter?")
```

`createdb`, `CREATE EXTENSION vector`, `pip install post-graph-rag`. That is the whole dependency list — plus any OpenAI-compatible endpoint, including a local one.

---

## Where the status quo sits

Two open-source systems define the current landscape.

**Microsoft GraphRAG** introduced the pattern most implementations now follow: LLM-driven entity and relation extraction, hierarchical community detection over the entity graph, and per-community summaries that answer corpus-level questions. Its indexing pipeline writes Parquet tables; vector stores are pluggable via a factory pattern, but the graph itself is not — there is no graph-storage seam to swap out.

**LightRAG** is the lighter, faster descendant. It introduced dual-level retrieval (low-level entity keywords, high-level thematic keywords) and, importantly, a genuine four-way storage abstraction: KV, vector, graph and document-status, each pluggable, with PostgreSQL, Neo4j, Milvus and others available as backends.

Both are good systems. Three structural gaps recur across them and across most Graph RAG code in the wild.

**Storage fragmentation.** Your graph, your embeddings and your application data live in different systems with different consistency models. A failed write can leave the graph and the vectors describing different worlds, and nothing detects it.

**Edge labels that cannot be queried.** LLM-extracted relation labels are free text. Measured on a real corpus, LightRAG produced 460 distinct edge keywords across 421 relations — *more than one label per edge on average* — with entries like "social contact" and "claimed influence". These describe an edge in prose. They do not support `WHERE relation_type = 'worked_with'`.

**No temporal model.** Relations are timeless assertions. Neither system carries validity intervals or any notion of one assertion superseding another. LightRAG does support document deletion and re-indexing (`adelete_by_doc_id`, `adelete_by_entity`), so corpora can be updated — but an updated corpus still yields a graph in which contradictory facts coexist as equals.

The architectural bet here is that PostgreSQL already solves the first problem, and that the other two are tractable given a transactional substrate.

---

## The single-datastore argument

pgvector provides HNSW approximate nearest-neighbour search. PostgreSQL provides transactions, foreign keys, JSONB and mature operational tooling. A property graph is two tables — vertices and edges — with a JSONB payload column.

Putting all three in one database buys you a single consistency model, one backup, one connection pool, and the ability to write a graph vertex and its embedding in the same transaction. The underlying storage layer, **[post-graph](https://github.com/crajah/post-graph)**, adds schema-per-tenant realms, sub-tenant "spaces", shadow audit tables recording every mutation with old and new row images, and append-only history tables — it is a graph database on PostgreSQL in its own right, written up separately in *[Introducing post-graph](https://crajah.github.io/post-graph/)*.

The cost is that you inherit PostgreSQL's write semantics, which turns out to matter for concurrency. More on that below.

A minimal end-to-end example — index a document, then ask a question that requires the graph:

```python
from post_graph_rag import GraphRAG, RAGConfig, DocumentMetadata, QueryParam

rag = GraphRAG(RAGConfig(
    model="gemini-3.6-flash",
    embedding_model="gemini-embedding-001",
    embedding_dim=1536,
    realm="research_kb",
    schema_per_realm=True,
))
await rag.initialize()
await rag.index_text(text, metadata=DocumentMetadata(document="babbage.txt"))

res = await rag.query("Who built the Analytical Engine?",
                      param=QueryParam(mode="mix", top_k=5))

print(res["answer"])                    # synthesised, with inline citations
print(res["retrieved_graph_triples"])   # the edges that supported it
```

`index_text` chunks with overlap, extracts entities and relations, embeds everything, and writes vertices and edges. `query` runs vector search over both documents and entities, traverses one hop from whatever it matched, and synthesises an answer. The rest of this article is about why each of those steps is harder than it looks.

---

## Entity resolution: the difference between a graph and a list

This is the single highest-leverage component, and the easiest to get wrong invisibly.

A naive extraction pipeline creates a vertex per mention. The consequence is subtle: everything still works, queries still return results, and the graph is quietly useless for its intended purpose. "Charles Babbage" extracted from document A and "Charles Babbage" extracted from document B become two disconnected vertices. Cross-document traversal — the entire justification for building a graph — cannot happen.

The first fix is resolution by canonical name within a `(realm, space)` scope, enforced by a unique index on `lower(payload->>'name')`. That handles exact repeats.

It does not handle real prose, where the same entity appears as `Babbage`, `Mr. Babbage`, `Charles Babbage` and `he`. So extraction is additionally asked to return **aliases** — every other surface form it observed:

```
Charles Babbage   aliases: ["Babbage", "Mr. Babbage"]
Ada Lovelace      aliases: ["Ada", "Ada Byron", "Augusta Ada King",
                            "Countess of Lovelace", "Lovelace"]
```

Resolution then tries canonical name first, then alias lookup against a JSONB array. Two policy decisions matter:

**The fuller name wins.** If `Babbage` is indexed before `Charles Babbage`, the vertex is promoted and the shorter form demoted to an alias. Order of ingestion does not determine the canonical name.

**A placeholder never overwrites specificity.** Triple endpoints that extraction did not return as full entities are created as bare `Concept` stubs. Without a guard, a stub silently degrades a properly typed entity recorded earlier.

Measured effect on a four-article corpus: standalone short-form vertices (`Babbage`, `Lovelace`, `Ada`, `Countess of Lovelace`) went from 6 to 0, and entities carrying aliases rose from 47 to 268 depending on the model.

Two categories of non-entity are rejected outright. **Pronominal references** — `he`, `his father`, `the company` — cannot resolve to a stable vertex; `his father` appeared as a real vertex in two documents before this was added. **Bare conjunctions** — `Ada Lovelace and Charles Babbage` — are two entities and a relation, not one entity.

A rule considered and deliberately made opt-in: rejecting possessive phrases like `Babbage's father`. On real prose the same pattern also catches `Ampère's force law` and `Menabrea's paper`, which are legitimately named things. Roughly half its hits were false positives, so it ships off by default.

---

## Predicate vocabularies: making edges queryable

Ask an LLM for relations and you get expressive, unusable variety. On one corpus, extraction produced **395 distinct predicates across 705 relations** — 74% used exactly once. Every edge is nearly unique. You can read that graph; you cannot query it.

Three mechanisms address this in layers.

**Normalisation.** Case, separators and leading tense auxiliaries are stripped, so `was_appointed_knight_of` and `appointed_knight_of` collapse to one predicate rather than two.

**A controlled vocabulary.** Supplying a preferred predicate list steers the model at extraction time and snaps morphological variants afterwards.

**An explicit synonym map** for wordings that differ genuinely rather than morphologically.

```python
RAGConfig(
    predicate_vocabulary=["created", "designed", "built", "worked_with",
                          "influenced", "studied_under", "member_of"],
    predicate_aliases={"collaborated_with": "worked_with",
                       "developed": "created"},
)
```

Measured on the same corpus: **395 → 44 distinct predicates across 417 relations, with 94% vocabulary adherence**, and a distribution with a genuine head — `located_in`(38), `worked_with`(36), `studied`(34). Before, the most common predicate appeared 6 times.

One subtlety worth stating because it is a trap: **the alias map must never collapse an inverse onto its converse.** Mapping `educated_by` onto `taught` reverses edge direction and manufactures false facts. Inverses get their own canonical predicate (`studied_under`), which limits how aggressively any vocabulary can compress.

A long tail of single-use predicates survives by design. `funded`, `lived_at`, `co_authored` are genuinely distinct relations that a 35-term vocabulary does not cover. Forcing them onto vocabulary terms would destroy information.

---

## Extraction quality: four smaller interventions

**Gleaning.** Single-pass extraction under-recalls on dense text. A follow-up prompt listing what was already found and asking only for what was missed raised relation counts roughly 40%. This mirrors GraphRAG's approach and costs one additional LLM call per chunk.

**Document context.** Each chunk is extracted with the document title, source and the canonical entity names discovered so far. Without it, every chunk after the first is extracted blind and its pronouns become junk vertices — this is where `his father` came from.

**Negation as a flag, not a predicate.** Models will happily emit `did_not_have_relationship_with` as a predicate. Traversal then treats a denial as a connection, and synthesis reads it as an assertion. Extraction instead uses the positive predicate with `negated: true`, and retrieval renders it explicitly as `NOT`.

**Relation provenance.** Each relation records the chunks that asserted it, and `weight` counts *distinct contributors* rather than write count. Without this, re-indexing a document three times produced `weight: 3` — making a single source look independently corroborated, which then fed ranking.

---

## Community summarisation

Some questions have no answer in any passage. "What are the main themes across this corpus?" is a property of the whole graph.

Following GraphRAG, the entity graph is clustered and each cluster summarised by an LLM. The implementation detail that matters here: **each report is stored as a vertex with its own embedding**, and membership edges link it back to its constituent entities. Global retrieval therefore finds themes by vector similarity rather than by enumerating relations, and every theme is traceable to the subgraph that produced it.

```python
await rag.build_communities()
res = await rag.query("What are the main themes?",
                      param=QueryParam(mode="global"))
```

**Clustering algorithm dominates output quality.** Measured on the same 428-entity graph:

| Detector | Largest community | Share of graph |
| :--- | ---: | ---: |
| Label propagation | 150 entities | 35% |
| Leiden, resolution 1.0 | 93 entities | 21% |
| Leiden, resolution 2.0 | 74 entities | **17%** |

A community holding a third of the graph is not a theme; its summary is a summary of everything. Leiden therefore ships as a core dependency, with a deterministic label-propagation fallback for platforms where the native `igraph` build is unavailable. Determinism is load-bearing — a randomised partition produces a different graph on every indexing run.

**Ranking reports needs care.** Ranking purely by embedding distance answers "what are the main themes?" from a niche corner of the graph, because a narrow cluster's summary sits closer to a short, broad query precisely *because* it covers less ground. Ranking now blends similarity with the report's self-assessed importance and log-scaled size.

The first implementation min-max normalised those signals across candidates, which is wrong: over a handful of candidates, min-max stretches a 0.04 cosine gap into the full range, so a negligible similarity edge dominates everything else. Absolute scales — `1 − distance`, `rating / 10`, `log1p(size)` — keep a decisive similarity margin decisive and a trivial one trivial.

---

## Concurrency, and what must not be parallel

Indexing was initially 6.4× slower than LightRAG on identical input because chunks were processed strictly sequentially. Indexing is almost entirely network-bound, so this was pure waste.

Each chunk now splits into a **prepare** phase — extraction, gleaning, embeddings, touching no database state — and a **write** phase. Prepare runs concurrently in bounded batches; writes are applied in order.

The serialisation is deliberate and worth dwelling on, because it is the cost of the transactional substrate. Entity resolution is a read-modify-write against a uniqueness index: read the existing entity, merge aliases and descriptions, write back. Concurrent writers race and can split entities that should have merged — reintroducing exactly the defect the resolution logic exists to prevent. Since writes cost microseconds against local PostgreSQL while LLM calls cost seconds, serialising them is nearly free.

Coreference context is threaded at **batch granularity**: a chunk sees entities discovered by all earlier batches, but not by its batch-mates. Full per-chunk threading is inherently sequential; this is an explicit trade, recoverable by setting concurrency to 1.

Result: **12.7 minutes → 3.7 minutes** on the same corpus, producing an equivalent graph (523 vs 488 entities, 671 vs 705 relations, 24 vs 26 orphans).

---

## Temporal evolution

Corpora change. Documents get revised. Facts expire. Two characters are allies in one book and enemies in the next.

**Re-indexing.** Documents carry a stable key (source, falling back to title) and each chunk a content hash. Re-indexing skips unchanged content, replaces changed content, and deletes removed documents — rather than appending a second copy, which previously duplicated chunks and inflated relation weight.

**Dormancy.** Entities orphaned by a deleted document are marked `dormant_since` rather than deleted, excluded from retrieval and community rebuilds, and revived if a later document mentions them. Deleting them would discard exactly the history the audit trail exists to preserve.

**Supersession.** Declaring mutually exclusive predicate groups lets a later assertion close an earlier one:

```python
RAGConfig(exclusive_predicate_groups=[{"friend_of", "enemy_of", "rival_of"}])
```

The superseded edge is hidden from retrieval by default and recoverable with `include_superseded=True`. The design choice that matters: **this resolves from document order, not from extracted dates.** Asking a model to reliably state when a relationship began is asking for precisely the thing models do least reliably — see the next section.

Tested on a genuinely temporal corpus: the d'Artagnan trilogy, three novels following the same characters across roughly forty years, indexed in publication order. From 350 chunks producing 1,377 entities and 3,976 relations, **13 relationships were closed by a later assertion** — among them an `ally_of` edge between d'Artagnan and Aramis, closed once a later volume recasts them as opponents, and Rochefort's `friend_of` link to the Cardinal, closed as his allegiance shifts.

Two findings worth carrying over to your own corpus. First, this worked with **no dates whatsoever** from the extractor on those pairs; publication order alone was sufficient — which is precisely why the design does not depend on models extracting dates. Second, density matters more than volume: an earlier run over the same novels sampled at roughly 5% fired zero times, because no entity pair was described twice. Supersession has nothing to act on until a relationship is mentioned more than once.

**Validity intervals**, where the text actually states them:

```python
QueryParam(as_of="1825")
```

The critical semantic: a relation with **no** stated period matches every `as_of` date. Silence about when a fact held means it held throughout — so a corpus that never mentions dates is entirely unaffected by as-of filtering. On a 701-relation corpus, 685 relations carried no stated period and `as_of` filtering correctly left them all in place.

---

## Multi-hop retrieval, and why depth is dangerous without ordering

One hop answers *what is said about X*. It cannot answer *how did X come to cause Y*, because the edges that carry that chain sit between X's neighbours and are never adjacent to X at all.

```python
RAGConfig(max_hops=2)                  # default since 1.4.0
QueryParam(max_hops=3)                 # per query
```

The filters travel *into* the walk rather than being applied to its result — `relation_types`, `as_of`, superseded-edge exclusion and space scoping all constrain every step. That distinction is not cosmetic. Filtering afterwards still lets a path travel *through* a closed or out-of-period edge to reach something that then presents as current context; the offending edge disappears from the output while the conclusion it enabled remains. Constraining each step means those paths are never walked.

Measured on the Boeing filings — relations retrieved per question, and how many mention a term from the question:

| Question | 1 hop | 2 hops | 3 hops |
| :--- | ---: | ---: | ---: |
| Deferred production costs arc | 7 (100% on-topic) | 30 (47%) | 202 (10%) |
| **What turned cash flow negative** | 13 (77%) | 20 (60%) | 59 (39%) |
| 737 ↔ cash flow over time | 32 (91%) | 125 (48%) | 435 (23%) |
| FAA role across filings | 31 (74%) | 73 (52%) | 307 (18%) |

**Precision collapses while absolute signal rises.** Three hops reaches three times as many on-topic relations and ten times as much noise. Recall alone is therefore the wrong thing to optimise — reaching more relations is trivial, reaching more *relevant* ones is the claim, which is why the table reports both.

**Depth pays exactly where the question is a chain.** *What caused Boeing's cash flow to turn negative* is the one question a single hop could not answer; it returned "the documents do not contain a direct statement specifically addressing" it. At three hops the same query answers directly, connecting the $11.3bn revenue decline, the January 2024 grounding of the 737-9 and the fixed-price development charges into one causal account. The other three questions answered at every depth.

**Ordering is what makes depth safe, and getting it wrong is subtle.** Relations were originally ranked newest-asserted-first, which is right at one hop and quietly wrong beyond it: assertion time across a corpus is close to arbitrary, so a three-hop edge that happened to be indexed last would displace an adjacent one the moment the relation token budget truncated. Ranking nearest-hop first, and only then newest-first within a hop, fixes it. The effect is visible in the results — two and three hops frequently synthesise *identical* answers, because the budget keeps the same nearest relations either way. Depth costs more; it no longer degrades.

Fan-out is the risk being managed here. Three hops from one well-connected Boeing entity reaches **over 25,000 edges**, which is noise rather than context, so `max_relation_edges` caps what any single matched entity may contribute.

The default is 2. Questions about a single entity are better served by 1, where precision is highest and the walk is cheapest.

---

## Retrieving relations by embedding, not only by traversal

Everything above ranks relations that traversal already reached. That framing hides an assumption worth testing: that the relations worth having are reachable from a matched entity at all.

They are not always. A relation whose endpoints are named generically — a charge, a programme, a line item that the filing never christens — sits in the graph and is never walked to, because nothing in the question matches its endpoints. No hop budget reaches it and no ranking recovers it.

That distinction is testable, so it was tested. Re-ranking the entire multi-hop candidate set by direct query similarity — a strictly better ranker over the same candidates — moved the on-topic share of retrieved relations from **67.8% to 67.5%**. Nothing. The bottleneck was never the ordering.

Generating candidates a second way is what pays. Relations carry their own embeddings, so they can be searched directly rather than walked to:

```python
RAGConfig(embed_relations=True,        # default since 1.5.0
          relation_seed_quota=0.5)     # share of slots for the similarity channel
```

| On-topic share of relations reaching the prompt | quota 0.0 | 0.5 | 1.0 |
| :--- | ---: | ---: | ---: |
| gpt-oss-120b | 49% | **69%** | 73% |
| MiniMax-M2.7 | 66% | **88%** | 98% |

Quota 0.0 is traversal alone, the behaviour before this change; 0.5 is the shipped default, in bold; 1.0 gives the similarity channel first claim on every slot.

The two channels are interleaved to a quota rather than pooled and ranked together, and that is not a stylistic choice. Pooling both candidate sets and sorting by a single similarity score does not split the difference — it hands *every* slot to the relation channel, because that channel is ranked by the very quantity being sorted on. Measured: a pooled ranking reproduced the relation channel's output exactly, on all four questions.

### The part where the obvious metric was wrong

Setting that quota needed a number, and the number that was easy to compute turned out to be untrustworthy.

Scoring retrieved relations by keyword overlap with the question rewards topical resemblance — which is precisely what embedding similarity maximises. The metric was therefore rewarding the channel under test. It ranked quota 1.0 above 0.5 on both models, and it would have said so regardless.

The tie-break was a blind comparison on answers instead: generate an answer at each setting through the shipped query path, then have judge models pick the better one. Three controls, each of which earned its place:

- **Blind.** The judge sees the question and two answers labelled A and B, and is told nothing about how either was retrieved.
- **No self-grading.** No judge model grades prose its own model wrote.
- **Order-swapped.** Every pair is graded twice with the labels swapped. A win counts only if the judge picks the same answer both ways.

That third control removed **18 of 50 judgements**, more than a third — 7 of 20 on one graph and 11 of 30 on the other — including a stretch where one judge answered "A" to everything put in front of it. A single-pass evaluation would have counted every one of those as a result.

What survived did not crown a winner. The two graphs pointed opposite ways overall. Split by question shape, they agree:

| Question shape | quota 0.5 | quota 1.0 |
| :--- | ---: | ---: |
| Entity — *what role did the FAA play* | **8** | 4 |
| Thematic — *what caused cash flow to turn negative* | **7** | 4 |
| Chain — *how did regulatory action translate into financial consequences* | 2 | **6** |

Traversal-weighted retrieval leads where the question names its subject or a theme. Similarity-weighted retrieval leads on chain questions, and led 5–0 on the weaker of the two graphs — traversal to depth accumulates noise faster when extraction is poorer, so the slots are better spent elsewhere.

The judges' written reasons explain why no single setting wins. Both directions are argued on identical grounds — concrete figures present, extraneous material absent — with 0.5 praised for "detailing concrete earnings charges and programs ($3.5B for 777X, $580M for 767)" in one comparison and 1.0 praised for "citing concrete examples like the $148 million Spirit litigation charge" in another. The criterion is stable; which setting satisfies it depends on the graph.

So it ships as a documented knob defaulting to 0.5, with the harness that produced this table in the repository. Turning the channel on at all is the unambiguous win — the 49%→73% and 66%→98% columns are not close. Where exactly to set the dial is a property of your corpus, and it is measurable in an afternoon.

---

## LongMemEval: the full benchmark, against Zep's published numbers

The mechanisms above are the design. This is the part that tests it against someone else's data, scored by someone else's method, on the benchmark Zep publish for Graphiti — the fairest available comparison for a temporally-aware graph.

On the full 500-question oracle set, all six question types, post-graph-rag with `gemini-3.7-flash` scores **68.3%**. Zep report **71.2%** with gpt-4o and **63.8%** with gpt-4o-mini; their full-context gpt-4o baseline is 60.2%.

| question type | n | post-graph-rag (flash) | Zep (gpt-4o) | Zep (gpt-4o-mini) |
| :--- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 91.4% | 92.9% | 81.4% |
| single-session-assistant | 55 | **89.1%** | 80.4% | 81.8% |
| knowledge-update | 78 | 70.5% | 83.3% | 76.9% |
| **multi-session** | 133 | **66.2%** | 57.9% | 40.6% |
| temporal-reasoning | 133 | 52.6% | 62.4% | 36.5% |
| single-session-preference | 30 | 50.0% | 56.7% | 30.0% |
| **overall** | 499 | **68.3%** | 71.2% | 63.8% |

Read against the comparable model tier, this is a win: a flash-class model beats Zep's gpt-4o-mini configuration on five of six categories and by 4.5 points overall, at 7.2s median query latency. Read against their best configuration, it is 2.9 points short — with the two exceptions worth naming. **Multi-session**, the category most demanding of cross-session synthesis and the one Zep themselves score lowest on with gpt-4o, is where post-graph-rag is furthest ahead (+8.3 over their gpt-4o). Single-session-assistant is the other (+8.7).

The comparison is not perfectly controlled, and the differences run in both directions: Zep judge with GPT-4o where this uses a three-model panel (MiniMax-M2.7, gpt-oss-120b, DeepSeek-V3.2, majority vote), and their generation models are a tier above `gemini-3.7-flash`. One question of 500 is excluded — a session both extraction prompts refused to extract — and the harness marks the run non-reportable until that count is zero, so it is stated here rather than absorbed.

### How the number was reached, and what that guards against

The configuration was frozen before the full run, after development against a 120-instance stratified sample on one seed. That discipline exists because the same code had already produced 75% on a favourable 20-instance draw — temporal-reasoning scored 81% on that sample and 52.6% on the full set. Sampling variance at small n is not a rounding concern; it is the difference between "beats Zep" and "does not".

Development was equally instructive for what it rejected. Four candidate improvements were tested on the dev seed; two survived. An answer-side rule resolving conflicting records to the most recent statement lifted knowledge-update by twenty points and survived. Wider retrieval (`top_k` 32) survived. A refinement distinguishing updates from accumulating preferences regressed the dev score and was dropped. Query decomposition for two-event questions — mechanistically the best-motivated of the four, built as a general engine feature — cost 6.7 points on the dev seed and was dropped from the configuration while remaining in the library, off by default. A held-out seed then showed the dev results themselves swung ±15–35 points per category at 12–31 instances per type, which is why the reported figure comes from the full set and nothing smaller.

The remaining deficit is concentrated and diagnosed: knowledge-update failures retrieve both the old and the new value and present the conflict — supersession not firing at indexing when the two statements extract into different entity-pair shapes — and temporal-reasoning failures hold one endpoint of an interval with the other never extracted with a resolvable date. Both point at indexing, not prompting.

Both indexing-side fixes were subsequently built and dev-tested, and neither survived — but the failure taught something worth more than either fix. Enabling LLM contradiction detection and broadening extraction to mundane dated actions are unrelated changes, yet they produced near-identical category swings: knowledge-update down 16–21 points, preference up 18, in both. That is not two features failing the same way; it is re-indexing itself reshuffling categories by more than any effect being hunted, on a benchmark where each instance's graph is rebuilt per run. The paired-ablation principle above — re-indexing per arm measures the extractor, not the feature — applies to its own follow-up work: indexing-side changes cannot be evaluated at 120 instances, and the honest cost of testing one is repeats per configuration or the full set per candidate. The 68.3% stands, and the two diagnoses remain open with their first proposed mechanisms measured and rejected.

### Three features adapted from Graphiti

Three retrieval features are adapted from Graphiti's design, all shipping off by default: **MMR** diversification of the merged candidates, **node-distance reranking**, and **LLM contradiction detection** to complement the declarative supersession above. Measured paired on a 20-instance slice — each instance indexed once, every variant answering from that identical graph:

| | accuracy | delta |
| :--- | ---: | ---: |
| baseline | 60% | — |
| MMR | 65% | +5 |
| **node distance** | **70%** | **+10** |

Contradiction detection changes what is written rather than what is read, so it gets its own re-indexed baseline: 75% against 70%. These ablations are relative comparisons on a small paired sample; the absolute numbers are not comparable to the full-benchmark table above.

### Pairing is what makes those numbers mean anything

Measured the obvious way, with a fresh index per variant, node-distance reranking scored −5. The same code scores +10 once every variant reads the same graph. The repeats explain it: they agreed in 80 of 80 cells, so with the graph held fixed the system is deterministic, and all the movement lives in extraction. A run that re-indexes per arm is measuring the extractor rather than the feature.

The same baseline has scored 75%, 70% and 60% on identical code and identical instances, purely from re-indexing. At 20 instances, effects of this size are promising rather than established — the sign test on node distance gives p = 0.31, and separating them cleanly needs 100–200 instances. All three point the same way, and each addresses a specific gap: MMR because RRF fuses three channels that correlate and a restatement costs a slot at top_k of eight; node distance because the relation-embedding and lexical channels report `hops=1` for everything, never having walked; contradiction detection because `exclusive_predicate_groups` only fires on predicate pairs declared in advance between the same two entities, and so cannot see "lives in Paris" becoming "lives in Berlin".

---

## Evaluation against LightRAG

LongMemEval measures answers. This measures the graph underneath them — density, resolution, and whether contradictions can be expressed at all — against the closest comparable library, on three corpora deliberately different in register: four Wikipedia articles (~66k chars), the d'Artagnan trilogy (~645k chars of 19th-century narrative prose), and five Boeing 10-K filings (~587k chars of financial disclosure). Same model (MiniMax-M2.7), same embedding model, same gleaning depth, equivalent chunk sizing, and — for the trilogy — the two libraries run **sequentially** so neither contends for the LLM endpoint.

Register turns out to matter more than corpus size — it shifts the speed gap, the density gap and the extraction failure modes, and each corpus below is chosen to show a different one.

### Encyclopedic prose

| | post-graph-rag | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| Characters indexed | 66,345 | 75,751 |
| Indexing time | 3.7 min | **2.0 min** |
| Entities per 10k chars | **78.8** | 59.4 |
| Relations per 10k chars | **101.1** | 55.6 |
| Distinct edge labels | **380** | 460 |
| Labels ÷ relations | **57%** | 109% |
| Query latency (mix / global) | **8.2s / 3.1s** | 10.1s / 6.1s |

LightRAG indexes roughly 2× faster. It parallelises across documents as well as chunks, and its in-process graph store has no transactional constraint to honour — a real advantage of that design, not an accident.

post-graph-rag produces a denser graph: 33% more entities and 82% more relations per unit of text.

### Narrative prose, where facts change

The trilogy is the harder and more interesting corpus: three novels, ~645k characters, the same characters across forty years, with alliances that genuinely reverse.

| | post-graph-rag | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| Indexing time | 33.2 min | **24.3 min** |
| Entities | **1,377** | 1,246 |
| Relations | **3,976** | 1,447 |
| Relations per entity | **2.9** | 1.2 |
| Distinct edge labels | 2,287 | 1,923 |
| Labels ÷ relations | **58%** | 133% |
| Entities carrying aliases | **729** | not modelled |
| **Relationships superseded** | **13** | **0** — unsupported |

Three things change on this register.

**The speed gap narrows**, from 2.1× to 1.37×. Long prose has fewer of the short, easily-parallelised documents that favour LightRAG's across-document concurrency.

**The density gap widens sharply**: 2.9 relations per entity against 1.2. Gleaning earns more on narrative text, where relationships are stated obliquely across paragraphs rather than declared outright.

**Edge labelling degrades further for free-text keywords.** LightRAG reaches **133%** — more than one unique label per relation, meaning on average *every edge carries its own label*. Novelistic phrasing is more varied than encyclopedic phrasing, so the distance between free text and normalised predicates grows exactly where you would least want it to.

And the row that has no counterpart: **13 relationships closed by a later book**, against a system with no way to express supersession at all. LightRAG's 1,447 relations contain the same contradictions, all sitting as equally current facts.

**Two methodological notes.** LightRAG's first Wikipedia run failed 3 of 4 documents on provider credit exhaustion and produced 113 entities in 34 seconds — which would have supported a "22× faster" claim from an entirely broken run. Its first trilogy run died the same way. Both completed only after the retry budget was raised; the successful trilogy run absorbed **261** HTTP 402 responses. Any benchmark that does not verify completion is not a benchmark, and on a key-rotating router a generous retry budget matters more than a fallback model list.

### Financial filings, where time is the whole question

The third corpus is the one where the temporal model earns its keep, and the one that stresses extraction hardest.

Five Boeing 10-K filings — Management's Discussion and Analysis only, 586,775 characters — chosen to span four operational cycles rather than sampled evenly, so relationships genuinely reverse between documents: the 777 boom in FY2006, the 787 supply-chain and battery crisis in FY2012, record free cash flow in FY2018, the 737 MAX grounding and negative cash flow in FY2020, and the quality and regulatory freeze in FY2024.

Annual filings are an unusually direct temporal corpus. The same line items recur every year while their significance inverts, and the filing says so in prose rather than in a dated field. The test question writes itself: *trace how deferred production costs evolve from a minor line item to the central driver of corporate cash burn.*

| | post-graph-rag | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| Entities | **3,078** | 2,832 |
| Relations | **4,830** | 3,147 |
| Relations per entity | **1.6** | 1.1 |
| Labels ÷ relations | **46%** | 77% |
| Relations with stated validity | **845** | not modelled |
| **Relationships superseded** | **8** | **0** — unsupported |

No indexing time appears in that table, deliberately. Both runs shared an LLM router with other work, so no wall-clock figure from this corpus is uncontended, and publishing one would be inventing precision. The trilogy timings above *were* run sequentially and do stand.

**The two systems answer the headline question differently in kind.** LightRAG returns a correct, well-written and entirely *timeless* account of what deferred production costs are and how the accounting works. post-graph-rag returns the trajectory, because the retrieved relations carry periods — and as-of filtering behaves accordingly: relation counts grow 22 → 24 → 25 across 2006 → 2024, and

```
(777X deferred production costs) --[reduced_by]--> (777X program)  [2020-12-31..open]
```

surfaces only when the question is asked as of 2024. A definition versus an arc. That distinction is the entire argument for modelling time in the graph rather than hoping the language model infers it from retrieved text, and no amount of retrieval tuning on an atemporal graph produces it.

The same asymmetry runs through the table. 845 relations carry a validity period lifted from prose that contains no dated field anywhere, and eight assertions were closed by a later filing. LightRAG's 3,147 relations contain the same reversals — the 737 generating cash in 2018 and consuming it in 2020 — sitting side by side as equally current facts, because there is no construct available to express that one superseded the other.

**Financial prose is the adversarial register for graph extraction**, and this is a finding about the genre rather than about either library. Encyclopedic and narrative text yield naturally canonical entities: people, places, works. Filings do not. Language models nominalise them, minting document-specific compound entities that can never recur:

```
Boeing Commercial Airplanes revenue increase
Boeing Company Q4 2005 net loss
737 programme production impacts
```

Each of those becomes a vertex, and each appears in exactly one filing. The graph fragments: a query for "cash flow" is smeared across dozens of hyper-specific vertices instead of landing on a few well-populated ones. Supersession starves for the same reason, since it fires only when one entity pair is characterised twice.

The correction is a single rule in the extraction prompt — emit the stable entity that could be named again in a *different* document, and put the movement, period and magnitude in the relation instead. `Boeing Commercial Airplanes`, not `Boeing Commercial Airplanes revenue increase`. The effect on retrieval, measured as relations returned per question:

| Question | before | after |
| :--- | ---: | ---: |
| Deferred production costs arc | 9 | 8 |
| What turned cash flow negative | **2** | **13** |
| 737 ↔ cash flow over time | **5** | **44** |
| FAA role across filings | 19 | **31** |

Recall rose between 2× and 9×, and supersession from 5 to 8. Worth stating plainly: before that rule, the two questions retrieving fewest relations were answered *"not directly addressed"*, while LightRAG answered both substantively. Retrieval recall, not generation, was the constraint — and it is the kind of defect that only a corpus of this register exposes.

A methodological note worth more than the fix. The metric chosen in advance to judge the change — the share of entity pairs carrying exactly one edge — did not move at all: 89.0% to 88.8%. Everything else improved regardless. Granularity did improve, and that is what lifted retrieval, by giving the query embedding a few dense vertices to land on. But pair recurrence turns out to be governed by the topical breadth of filings rather than by naming, because most subject-object pairs in a 10-K genuinely occur once however clean the names are. A plausible proxy metric, chosen before the evidence, would have condemned a change that worked.

Three narrower guards now sit in the extractor, all of them invisible on narrative text: entity names are length-capped, because a filing occasionally returns an entire table row as a name and the unique index behind entity resolution rejects it at 2,704 bytes; bare quantities such as `$18.4 billion` are refused as vertices, since a figure is the *value* of a relation rather than a thing that holds relations; and embedding requests state `encoding_format` explicitly, because the OpenAI SDK otherwise negotiates base64 and gateways fronting non-OpenAI providers reject the parameter outright, failing every embedding call and producing a silent empty graph.

**One caveat on vocabulary.** The `finance` preset supplies 32 predicates and 8 exclusivity groups, and every group member was emitted, so the controlled vocabulary works as designed. But labels sit at 46% of relation count here against the ~11% reached on biography — still well ahead of LightRAG's 77% on the same corpus, though the gap narrows. Filings discuss a far wider range of topics than a Wikipedia article, and 32 terms cover the reversals without covering the tail. Expect to extend the vocabulary per domain rather than inherit one.

---

## What an evaluation harness is for: three defects it found

Since 1.8.0 the ECT-QA harness — sixteen quarters of earnings calls per company, where the same metric is restated every quarter and only the date separates the values — has been less useful for the score it produces than for the faults it exposed. All three were invisible in normal use.

**A constant `source` was deleting documents.** `document_key()` preferred `source` over the document title and discarded the title entirely, so a caller passing a corpus name rather than a path collapsed every document onto one key. Since a matching key means *re-index*, each document deleted the one before it. An 80-transcript corpus retained five transcripts, 92% of relations were marked dormant, and nothing raised an error. The only symptom was the system answering "unanswerable" — which was correct, given what remained. The key now combines both parts; realms indexed earlier should be rebuilt, since their keys no longer match.

**The judge panel was scoring worse than the system.** Asked for gross margin in each quarter of 2022, the answer gave 33.9%, 33.6%, 31.7% and 32% against a gold of 33.9%, 33.6%, 31.7%, 32.3% — three exact, the fourth within 0.3 points. All three judges failed the whole answer. Scoring moved to tolerance-based numeric F1, and rescoring identical answers took the run from 0.162 to 0.265. Ten points that were never a system failure.

Cosine similarity was the obvious replacement and was rejected on measurement: gold is a bare list of figures while the system replies in prose, so whole-text embedding is dominated by length. Correct answers averaged 0.643 and refusals 0.509 — 0.13 apart, with no threshold between them. Figure matching separated the same rows 0.770 against 0.144. Cosine remains the fallback for questions whose gold carries no figure.

**The answer prompt was instructing the refusals.** It ended *"if the facts do not support an answer, say exactly: unanswerable"*, and the model read that as requiring completeness — refusing 45 of 60 questions, several while quoting figures it had already retrieved. Under F1 a partial answer earns partial credit, so the wording was discarding points the metric would have awarded. Rewritten to ask for partial answers, refusals fell from 75% to 7% and mean numeric F1 rose from 0.201 to 0.419. `top_k` also moved from 12 to 48, since gold answers need a mean of 5.5 figures and up to 32 — usually one per quarter across sixteen quarters, which twelve chunks cannot cover however well they are ranked.

A fourth change followed from watching what the third cost. Encouraging partial answers helped multi-period questions and hurt single-period ones — the model volunteered neighbouring quarters where exactly one figure was wanted, and precision in the F1 charged it for them. Adding scope discipline — answer the periods asked and no others, and when no period is named answer for the most recent the facts cover — recovered that without giving back the gain.

| | accuracy | mean numeric F1 | refused |
| :--- | ---: | ---: | ---: |
| judge panel, original prompt | 0.162 | — | 75% |
| numeric F1, same answers | 0.221 | 0.191 | 75% |
| + extraction split, `top_k` 48, partial answers | 0.324 | 0.447 | 7% |
| **+ scope discipline** | **0.353** | **0.451** | 10% |

These figures were themselves corrected once. The first numeric-F1 implementation counted every number in the text as a figure, which cut both ways: a prose answer citing "fiscal 2023-q2 [1]" bled precision on chronology it never asserted as a value — single-time questions containing their gold figures scored 0.38–0.59 against a 0.60 threshold, reading as a category at zero — while gold answers stating a year alongside each figure handed out free recall matches, inflating multi-time. Chronology and citation markers are now excluded from figure matching, and questions whose gold *is* a period ("Q1 2022") get period-token matching of their own instead of falling to cosine. The corrected column is lower than the first published version of this table, and the correction is stated rather than silently applied.

By question type under the corrected metric: single-time 0.421 — the earlier "category at zero" was the metric artifact above, not the system. Multi-time 0.280 and relative-time 0.167, both previously overstated by year-matching. Cross-company 0.100: answers now produce figures, but frequently for a different quarter than the question intends, which is a real failure of period selection rather than of retrieval or scoring.

Cross-company remains **0.000 over ten questions**, unmoved by every change so far, and two hypotheses for it have now been falsified rather than confirmed. It is not retrieval: all four companies are retrieved even at `top_k=12`, with eleven chunks discussing the metric asked about. It is not the all-or-nothing refusal wording either, since that fix moved every other category and left this one untouched. Ten questions at zero out of sixty is the part of this benchmark still unexplained, and an average that quietly absorbed it would be the more flattering number and the less useful one.

---

## Model sensitivity is larger than library differences

Holding corpus, settings and library fixed, and varying only the extraction model:

| | Llama-3.3-70B | DeepSeek-V3.2 | MiniMax-M2.7 | gemma-4-31B |
| :--- | ---: | ---: | ---: | ---: |
| Relations | 449 | 599 | **705** | 417 |
| Entities carrying aliases | 47 | 139 | **268** | 120 |
| Negated relations captured | **0** | 21 | 7 | 14 |
| Distinct predicates | 240 | 279 | 395 | **44** |
| Relations on vocabulary | 44% | 41% | 33% | **94%** |

The spread is the point. MiniMax builds the richest graph; gemma builds the most *queryable* one — 94% vocabulary adherence against 33%, an order of magnitude tighter predicate set, for 40% fewer relations. Density and queryability are separate goods, and post-graph-rag is built to be steered toward either: the vocabulary, gleaning depth and exclusivity groups are configuration, not assumptions baked into the pipeline. A system that hard-codes free-text edge labels cannot make this trade at all — whatever the model hands back is what the graph gets.

The `negated` column is the important one. **Llama-3.3-70B never populated that field once across an entire corpus**, silently collapsing "X worked with Y" and "X never met Y" into the same edge.

This generalises beyond one library: **any capability that depends on a model doing something subtle degrades silently on weaker models.** Alias merging only works if the model emits aliases. Validity intervals only work if it extracts dates. The machinery was byte-identical across all four columns; only the input differed. If you put an optional field in an extraction schema, measure whether your model actually fills it — nothing else will tell you.

---

## An aside on where Graph RAG defects hide

Everyone agrees systems should surface errors. What is less obvious is *where* Graph RAG specifically tends to bury them, and it is worth naming the pattern because it recurs.

Graph RAG defects are almost never visible at write time. They surface as a plausible, wrong answer at read time, often much later.

Three concrete examples from this codebase, all of which reported success while corrupting the graph:

**A heuristic extraction fallback.** When the LLM was unreachable, an earlier version chained adjacent capitalised words with a `relates_to` predicate, producing edges like `(Mount) --relates_to--> (Olympus)`. It returned `triples_extracted: 8`. Once stored, those edges are indistinguishable from genuine extracted structure, so a transient outage permanently degrades the graph with no marker.

**Weight inflation on re-index.** Relation `weight` counts corroborating sources and feeds ranking. Because re-indexing appended rather than replaced, indexing the same document three times produced `weight: 3` — one source presenting as three independent ones.

**Non-deterministic fallback embeddings.** The local fallback embedder used Python's `hash()`, which is salted per process. Same text, different vector on every run, so nothing indexed could be retrieved after a restart.

None of these raise. All produce confident answers. The general lesson is that in a probabilistic pipeline, the dangerous failures are the ones that still return a result — which is why every degradation path here either raises or is measurable, and why the useful distinction turned out to be that *skipping one bad chunk is recovery, skipping every chunk is an outage*.

---

## Limitations

**Community summarisation is single-level.** GraphRAG's hierarchical levels are not implemented, so semantically overlapping clusters can coexist. Leiden at resolution 2.0 keeps the largest cluster to 17% of the graph, which makes this tolerable rather than solved.

**Entity resolution is name and alias based.** There is no embedding-space clustering of near-duplicates, so `Science Museum` and `London Science Museum` remain distinct. Canonical-name and alias matching covers the common cases; near-duplicate clustering is the obvious next increment.

**Honorific-heavy prose stretches it further.** `M. d'Artagnan the younger`, `Monsieur d'Artagnan` and `D'Artagnan` remained distinct vertices across the trilogy, and periphrases like "the son of Henry IV" for Louis XIII are not resolved at all. 729 of 1,377 entities did carry aliases, so the mechanism works on that register — it is not exhaustive there. No system in this comparison models aliases at all, so this is a limit of an existing advantage rather than a missing capability.

**Extraction quality is register-sensitive, and a domain vocabulary is worth budgeting for.** Filings provoke nominalisation — `Boeing Commercial Airplanes revenue increase` as a vertex — which fragments retrieval until the extraction prompt is constrained. Predicate adherence moves with register too. With a vocabulary supplied, the distinct-label ratio is ~11% on biography but 46% on 10-Ks — a 32-term finance preset covers the reversals without covering the topical tail. Both still sit well below free-text edge keywords on the same corpora (109% and 77%), but the margin is far wider on encyclopedic text, and a vocabulary tuned per domain is what closes it.

**The retrieval quota is a knob, not a solved value.** Adding the relation-similarity channel is unambiguous — the on-topic share rises sharply on both models tested. How much of the context budget it should claim is not: on two graphs built from the same corpus by different extraction models, the blind comparison pointed in opposite directions overall, and only agreed once split by question shape. Two models and ten questions is a small basis for a default. The harness ships so the value can be measured per corpus rather than inherited.

**Supersession needs corpus density to fire.** It resolves only when the same ordered entity pair is characterised twice with conflicting predicates. On a sparse 5% sample of each novel that never happened and the mechanism sat idle; at full density it fired 13 times. A corpus that mentions each relationship exactly once has nothing to supersede — though it also has no contradiction to get wrong.

**Indexing is slower than LightRAG's, and deliberately so.** Graph writes are serialised because entity resolution is a read-modify-write against a uniqueness index, and concurrent writers split the entities that resolution exists to merge. That serialisation is what makes `Babbage` and `Charles Babbage` reliably converge — the density advantage in every table above depends on it. The gap is register-dependent, narrowing from 2× on short encyclopedic documents to 1.37× on long prose, and parallelising across documents would close much of the remainder without touching the write path.

---

## Try it

```bash
pip install post-graph-rag
createdb mydb && psql -d mydb -c "CREATE EXTENSION vector;"
```

The repository ships the full evaluation harness used for every number above:

```bash
python evaluation/fetch_corpus.py --series dumas --out evaluation/dumas
python evaluation/temporal_eval.py --corpus evaluation/dumas --realm dumas_kb \
    --model MiniMax-M2.7 --max-chunks 0 --max-concurrent-chunks 6
```

Or, for the financial corpus — a decade of SEC filings fetched straight from EDGAR:

```bash
python evaluation/fetch_sec.py --company boeing --out evaluation/sec_boeing
python evaluation/temporal_eval.py --corpus evaluation/sec_boeing \
    --realm boeing_kb --preset finance --model MiniMax-M2.7 \
    --max-chunks 0 --synthesise
```

That reports supersessions, validity intervals, dormant entities and as-of retrieval on a corpus whose facts genuinely change. Alongside it: corpus fetching, indexing with configurable models and vocabularies, a graph analysis report covering cross-document resolution and predicate distribution, a realm-diffing tool, the blind A/B judge used to set the retrieval quota, and the LightRAG comparison script. Every table here is reproducible rather than asserted.

A paper covering the architecture and evaluation methodology is going to arXiv — link to follow.

**GitHub:** https://github.com/crajah/post-graph-rag · **PyPI:** `pip install post-graph-rag`

Apache 2.0 licensed, built on PostgreSQL and pgvector, with multi-tenant realms, audit tables and append-only history — because a knowledge graph you cannot isolate, roll back or explain is not one you can put into production.

If there is one idea worth taking away independent of this library: **a graph whose edges are free text and whose facts never expire is a graph you can read but not query.** Both are fixable, and neither fix requires the model to do anything it is unreliable at.
