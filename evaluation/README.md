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

### Embedding models behind a gateway

Pointing the harness at a non-OpenAI provider — Vertex AI, Bedrock and friends,
usually via a proxy such as litellm — needs one setting. The OpenAI SDK
negotiates `encoding_format="base64"` on its own when the caller says nothing,
and gateways that front another provider tend to reject the parameter outright
rather than ignore it, which fails *every* embedding call:

```
litellm.UnsupportedParamsError: vertex_ai does not support parameters:
{'encoding_format': 'base64'}
```

`RAGConfig.embedding_encoding_format` states it explicitly and defaults to
`float`, the API default and the portable choice. Set it to `None` to restore SDK
negotiation for an endpoint that prefers base64. `compare_lightrag.py` honours
`RAG_EMBEDDING_ENCODING_FORMAT` for the same reason.

Note the failure mode rather than the fix: every chunk is skipped, the run
completes, and the graph is empty. Each document reports `FAILED EmbeddingError`
and the summary reads `entities=0` — so it is visible, but only if the output is
read. Check the entity count before trusting a fast run.

Embedding dimensions matter too. `--embedding-dim` must match what the model
actually returns, and the vector column is fixed at realm-creation time, so
changing embedding model means a new realm rather than a re-index. Gateways may
also reduce a model's native width, so probe rather than assume:

```bash
curl -s $OPENAI_API_BASE/embeddings -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-embedding-model","input":"probe","encoding_format":"float"}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))'
```

Worth checking before a long run: pgvector's HNSW index caps at 2000 dimensions,
so a model wider than that needs the index dropped or the width reduced.

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

That comparison uses encyclopedic prose. The two sections below repeat it on
corpora whose facts change — a novel sequence and a decade of annual filings —
where supersession and as-of retrieval have no LightRAG counterpart, and where
extraction behaves quite differently by register.

## Temporal comparison on a corpus that changes

`temporal_eval.py` indexes a corpus whose facts genuinely change. The bundled
example is the d'Artagnan trilogy — three novels following the same characters
across roughly forty years, during which alliances reverse.

```bash
python evaluation/fetch_corpus.py --series dumas --out evaluation/dumas
python evaluation/temporal_eval.py --corpus evaluation/dumas --realm dumas_kb \
    --model MiniMax-M2.7 --max-chunks 0 --max-concurrent-chunks 6
```

Books carry a numeric filename prefix so a plain sort gives publication order,
which is what supersession resolves from.

Head to head against LightRAG — same three books (~645k chars), same model
(MiniMax-M2.7), same embeddings, publication order, run **sequentially** so the
wall-clock figures are uncontended:

| | post-graph-rag | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| Indexing time | 33.2 min | **24.3 min** |
| Entities | **1,377** | 1,246 |
| Relations | **3,976** | 1,447 |
| Relations per entity | **2.9** | 1.2 |
| Distinct edge labels | 2,287 | 1,923 |
| Labels ÷ relations | **58%** | 133% |
| Entities carrying aliases | **729** | not modelled |
| Relations with stated validity | **83** | not modelled |
| **Relationships superseded** | **13** | **0** — unsupported |

**Supersession fires on real narrative prose.** Thirteen relationships were
closed by a later book, including several the trilogy is built on: an `ally_of`
edge between d'Artagnan and Aramis closed once a later volume recasts them as
opponents, and Rochefort's `friend_of` link to the Cardinal closed as his
allegiance shifts. All resolved from publication order alone — the extractor
supplied no dates for those pairs.

Density is the precondition. An earlier run sampling ~5% of each novel fired
**zero** supersessions, because no entity pair was characterised twice. Sampling
evenly but sparsely does not help; the same pair must actually recur.

**Edge labelling inverts here.** LightRAG reaches 133% — more than one unique
label per relation, meaning on average every edge carries its own label. Long
narrative prose produces far more varied phrasing than encyclopedic text, so the
gap between free-text keywords and normalised predicates widens.

**A caveat on entity resolution.** Fiction with heavy honorific variation is
harder than Wikipedia: `M. d'Artagnan the younger`, `Monsieur d'Artagnan` and
`D'Artagnan` remain distinct vertices, and `The Son of Henry IV` is a periphrasis
for Louis XIII. 729 entities do carry aliases, so the mechanism works — it is not
exhaustive on this register.

**On router stability.** The successful LightRAG run absorbed 261 HTTP 402
responses through retry. On a router that rotates API keys, 402 is transient
rather than terminal, and a generous retry budget matters more than a fallback
model list — an earlier attempt with only 5 retries died partway through.

## Temporal comparison on financial filings

`fetch_sec.py` downloads annual filings from SEC EDGAR. Annual reports are an
unusually direct temporal corpus: the same line items recur every year while
their significance inverts, and the filings say so in prose rather than in a
dated field.

```bash
python evaluation/fetch_sec.py --company boeing --out evaluation/sec_boeing
python evaluation/temporal_eval.py --corpus evaluation/sec_boeing \
    --realm boeing_kb --preset finance --model MiniMax-M2.7 \
    --max-chunks 0 --max-concurrent-chunks 6 --synthesise
```

Five Boeing 10-K filings, Management's Discussion and Analysis only, 586,775
characters. MD&A carries the mandated discussion of cash-flow drivers and
programme charges; the head of a filing is business description and risk
factors, which barely change between years. Fiscal years span four operational
cycles rather than being sampled evenly, so relationships genuinely reverse:

| Order | Filing | Cycle |
| ---: | :--- | :--- |
| 1 | FY2006 | record deliveries, strong margins |
| 2 | FY2012 | 787 supply chain and battery crisis |
| 3 | FY2018 | record revenue and free cash flow |
| 4 | FY2020 | 737 MAX grounding, negative cash flow |
| 5 | FY2024 | quality crisis and FAA production caps |

Term distribution tracks the arc: `737 max` peaks at 59 occurrences in FY2020,
`grounding` at 19, `faa` climbs to 10 by FY2024, `deferred production` appears in
all five.

Head to head, same model (MiniMax-M2.7), same embeddings, same chunk sizing,
whole documents:

| | post-graph-rag | LightRAG 1.5.6 |
| :--- | ---: | ---: |
| Entities | **3,078** | 2,832 |
| Relations | **4,830** | 3,147 |
| Relations per entity | **1.6** | 1.1 |
| Distinct edge labels | **2,203** | 2,413 |
| Labels ÷ relations | **46%** | 77% |
| Relations with stated validity | **845** | not modelled |
| **Relationships superseded** | **8** | **0** — unsupported |

Indexing time is deliberately absent: both runs shared a router with other work,
so no wall-clock figure here is uncontended. The Dumas table above was run
sequentially and its timings do stand.

### Financial prose is the adversarial case

Every earlier corpus is narrative, where entities are naturally canonical —
people, places, works. Filings are not. The model nominalises, minting
filing-specific compound entities that can never recur:

    Boeing Commercial Airplanes revenue increase
    Boeing Company Q4 2005 net loss
    737 programme production impacts

This fragments retrieval, because a query for "cash flow" is spread across dozens
of hyper-specific vertices instead of landing on a few well-populated ones. It
also starves supersession, which only fires when the same entity pair is
characterised twice.

Constraining entity granularity in the extraction prompt — emit the stable entity
that could be named again in a different document, and put the movement in the
relation description — raised supersession from 5 to 8 and lifted retrieval
recall between 2x and 9x. `%boeing%` vertices fell from 54 to 37.

Two narrower guards matter on this register and are now enforced in the
extractor: entity names are capped in length, because a filing occasionally
returns a whole table row as a name and the unique index backing entity
resolution rejects it; and bare quantities (`$18.4 billion`, `$1,326`) are
refused as vertices, since a figure is the value of a relation rather than a
thing that holds relations.

### Retrieval recall drives answer quality

Relations retrieved per question, before and after the granularity fix:

| Question | before | after |
| :--- | ---: | ---: |
| Deferred production costs arc | 9 | 8 |
| What turned cash flow negative | 2 | **13** |
| 737 ↔ cash flow over time | 5 | **44** |
| FAA role across filings | 19 | **31** |

Before the fix, the two questions that retrieved fewest relations were answered
"not directly addressed" — a retrieval failure rather than a generation failure,
and a straightforward loss against LightRAG, which answered both substantively.
After the fix all four answer, and the 737 question produces an era-by-era
trajectory.

On the headline test — *trace how deferred production costs evolve from a minor
line item to the central driver of cash burn* — both systems answer, but
differently. LightRAG returns a correct and **timeless** accounting definition.
post-graph-rag returns the trajectory, because the retrieved relations carry
periods. As-of filtering behaves accordingly: relation counts grow 22 → 24 → 25
across 2006 → 2024, and

    (777X deferred production costs) --[reduced_by]--> (777X program)  [2020-12-31..open]

surfaces only at `as_of=2024`.

### Vocabulary adherence has a longer tail here

The `finance` preset supplies 32 predicates and 8 exclusivity groups. Every group
member was emitted, so the vocabulary works:

    generates_cash_flow  80     consumes_cash      8
    increases_revenue    52     reduces_revenue   56
    incurs_charge        49     reports_loss      47
    improves_margin      12     erodes_margin      6
    certifies             4     grounds            3

But labels sit at 46% of relation count rather than the ~11% reached on
biography. Filings discuss an enormously wider range of topics than a
Wikipedia article, and a 32-term vocabulary covers the reversals without covering
the tail. Expect to extend the vocabulary per domain rather than to inherit one.

### Bounded multi-hop retrieval

One hop answers "what is said about X". Chain questions need the edges *between*
X's neighbours, which are never adjacent to X. `RAGConfig.max_hops` (or
`QueryParam(max_hops=...)`) walks further; it defaults to 2.

Filtering happens inside the walk rather than afterwards — `traverse()` takes
`relation_types`, `as_of`, `payload_null_keys` and `space` — so a path is never
routed *through* a superseded or out-of-period edge to reach something that then
looks current. Filtering only the result set would leave exactly those laundered
paths behind.

Measured on the Boeing corpus, MiniMax-M2.7, relations retrieved per question and
how many mention a term from the question:

| Question | 1 hop | 2 hops | 3 hops |
| :--- | ---: | ---: | ---: |
| Deferred production costs arc | 7 (100% on-topic) | 30 (47%) | 202 (10%) |
| What turned cash flow negative | 13 (77%) | 20 (60%) | 59 (39%) |
| 737 ↔ cash flow over time | 32 (91%) | 125 (48%) | 435 (23%) |
| FAA role across filings | 31 (74%) | 73 (52%) | 307 (18%) |

**Precision falls steeply while absolute signal rises.** Three hops reaches three
times as many on-topic relations and ten times as much noise. Recall alone is
therefore the wrong thing to optimise: reaching more relations is trivial,
reaching more *relevant* ones is the claim.

**Depth pays where the question is a chain.** *What caused Boeing's cash flow to
turn negative* is the question 1 hop could not answer — it returned "the
documents do not contain a direct statement specifically addressing" it. At three
hops the same query answers directly, with the revenue decline, the 737-9
grounding and the fixed-price charges connected into one causal account. The
other three questions answered at every depth.

**Ordering is what makes depth safe.** Relations are ranked nearest-hop first and
only then newest-first, so when the relation token budget truncates it sheds the
most tenuous connections rather than whichever edge happened to be indexed last.
Without that, ranking by assertion time alone let a three-hop edge displace an
adjacent one — assertion time across a corpus is close to arbitrary. With it, two
and three hops frequently synthesise identical answers, because the budget keeps
the same nearest relations either way: deeper costs more but does not degrade.

`max_relation_edges` (default 200) caps what a single matched entity may
contribute, which matters more than it sounds — three hops from one well-connected
Boeing entity reaches over 25,000 edges.

Start at 2 for chain-heavy corpora; leave it at 1 when questions are about a
single entity, where precision is highest and the walk is cheapest.

### Retrieving relations by embedding, not only by traversal

Traversal can only rank what it reached. A relation whose endpoints are named
generically is unreachable from a matched entity at any hop budget, and no
amount of re-ranking recovers it — re-ranking the multi-hop candidate set by
query similarity moved the on-topic share from 67.8% to 67.5%, which is nothing.

Embedding the relations themselves and searching them directly is a second
candidate generator rather than a better ranker, and that is where the gain is.
`embed_relations` (on by default) costs one embedding call per distinct triple at
index time; `relation_seed_quota` sets how many relation slots it may claim.

Mean on-topic share of the relations reaching the prompt:

| | quota 0.0 | 0.5 | 1.0 |
| :--- | ---: | ---: | ---: |
| gpt-oss-120b | 49% | 69% | 73% |
| MiniMax-M2.7 | 66% | 88% | 98% |

A quota is needed because pooling both candidate sets and sorting by one score
does not split the difference — it hands *every* slot to the relation channel.
Measured: a pooled ranking reproduced the relation channel's output exactly on
all four questions.

#### Why the quota is not simply set to 1.0

The scoring above counts keyword overlap, which correlates with the similarity
the relation channel ranks by, so it cannot be trusted to order 0.5 against 1.0 —
it rewards the channel under test. Settling those two needs answers.

```bash
python evaluation/blind_quota_judge.py --realm boeing_mm --model MiniMax-M2.7 \
  --judges gemma-4-31B-it DeepSeek-V3.2 --out reports/blind_quota_mm.json
```

It generates an answer at each setting through the shipped query path, then has
judge models pick the better one. Three controls: the judge
sees only the question and two answers labelled A and B; no judge grades prose
its own model wrote; and every pair is graded twice with the labels swapped, so a
judge that simply prefers the first answer produces no result rather than a wrong
one. That control mattered — **7 of 20 judgements on one graph and 11 of 30 on
the other were order-dependent and discarded.**

Wins over ten questions on two graphs. Judges were `gemma-4-31B-it` and
`DeepSeek-V3.2` on both, with `gemini-3.6-flash` added on the second:

| | 0.5 wins | 1.0 wins |
| :--- | ---: | ---: |
| `boeing_mm` (MiniMax-M2.7) | 10 | 3 |
| `boeing_oss` (gpt-oss-120b) | 7 | 11 |

The two graphs disagree, and the aggregate (17–14) hides why. Split by question
shape:

| shape | 0.5 | 1.0 |
| :--- | ---: | ---: |
| entity — *"what role did the FAA play"* | 8 | 4 |
| thematic — *"what caused cash flow to turn negative"* | 7 | 4 |
| chain — *"how did regulatory action translate into financial consequences"* | 2 | 6 |

0.5 leads where the question names its subject or a theme. 1.0 leads on chain
questions, and did so 5–0 on the weaker graph: traversal to depth collects noise
faster when extraction is poorer, so slots are better spent on similarity there.

The judges' stated reasons are worth reading, because they do *not* split by
setting. Both directions are argued on the same two grounds — concrete figures
present, extraneous material absent:

> *0.5 wins:* "more specific, detailing concrete earnings charges and programs
> ($3.5B for 777X, $580M for 767)"; "includes irrelevant information about 401(k)
> funding strategies" (against 1.0)
>
> *1.0 wins:* "more specific, citing concrete examples like the $148 million
> Spirit litigation charge"; "avoids extraneous details like the 777X loss"
> (against 0.5)

So the judges are applying a stable criterion, and what varies is which setting
happened to put more anchored specifics in front of the model. That is a property
of the graph, not of the quota — which is the honest reading of two graphs
pointing opposite ways.

**0.5 is the default**, as the setting that leads on the two commonest question
shapes and is never far behind on the third. Raise it toward 1.0 for a corpus of
chain questions, or where extraction quality is known to be poor. Anything below
0.5 gives up a large, unambiguous gain; the choice between 0.5 and 1.0 is worth
measuring on your own corpus, and this harness is how.

### Two harness settings that silently invalidate the comparison

`compare_lightrag.py` truncates each document to 20,000 characters by default,
which reduced this 586k corpus to 100k in an early run. Pass `MAX_CHARS=0`. It
also carries a fallback model list, which swaps models mid-run; pass
`RAG_FALLBACK_MODELS=` to hold one model, matching post-graph-rag.

```bash
CORPUS=evaluation/sec_boeing MAX_CHARS=0 EVAL_QUESTIONS=finance \
RAG_MODEL=MiniMax-M2.7 RAG_FALLBACK_MODELS= \
    python evaluation/compare_lightrag.py
```

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


## Bi-temporal storage, and a lexical retrieval channel

Two changes taken from Graphiti/Zep (arXiv:2501.13956) after comparing designs.

### Both temporal axes, kept separate

Relations previously recorded only *validity* — when a fact held in the world,
lifted from the prose. They now also carry **transaction time**: `t_created`
when this system came to believe the fact, `t_expired` when it stopped.

The two are independent, and conflating them loses a question. A 2024 filing
can assert something true in 2019, so:

- `as_of=2019` — what was true in 2019, according to everything known now.
- `as_believed_at=2023-01-01` — what the graph held on that date, regardless of
  when the facts themselves were true.

Only the second reproduces a past answer or audits a past decision. Supersession
now stamps `t_expired` as well as `superseded_by`, so a retracted belief stops
being current without ceasing to exist — a superseded row is still the correct
answer to a question asked about an earlier belief state.

A relation with no `t_created` predates the field and is treated as always
known, matching the existing rule that absent validity means always valid: the
absence records that nothing was said, not that the answer is no.

### A third candidate generator

`search_relations_text` adds lexical retrieval over relations using
PostgreSQL's own full-text search — a GIN index, no new infrastructure.

The motivation is the measurement already in this document: adding the
relation-embedding channel moved on-topic share 49% → 73%, while re-ranking the
same candidates moved it 67.8% → 67.5%. **Candidate generation is the
bottleneck, not ordering**, so a third generator is worth more than a fourth
ranker.

Lexical search is weakest where embeddings are strongest and strongest where
they are weakest: rare identifiers that carry the meaning but sit nowhere
useful in vector space — a part number, a statute, a designation like `737-9`.
Those are frequently the term a question turns on, and nearest-neighbour search
will return semantically adjacent relations mentioning none of them.

### Not adopted

**LLM-judged contradiction.** Graphiti asks a model whether two edges conflict.
Supersession here is decided by declared `exclusive_predicate_groups` and
document order — deterministic, free at write time, and inspectable. Their
approach catches undeclared conflicts; this one cannot be wrong in a way you
cannot audit. Given that model choice already dominates every result in this
document, putting a model in the resolution path is the wrong trade.

**Incremental community updates.** Their paper reports that dynamic label
propagation drifts from a full run and needs periodic refreshes. A Leiden
rebuild is a known-good partition, which is the better default until
incremental update is a measured bottleneck rather than an assumed one.


### Reciprocal rank fusion, and why the lexical channel needs it

The quota needs a constant saying how much context each channel deserves, and
the blind comparison could not settle it — 0.5 led on entity and thematic
questions, 1.0 on chain questions, and the two graphs disagreed overall. RRF
needs no such constant: a relation ranked well by several channels outranks one
ranked well by a single channel, so agreement does the work the quota guesses.

Measured on `boeing_mm`, mean on-topic share of the 25 relations reaching the
prompt. Every column receives identical channels, so the differences are the
merge and the channel, separated:

| | mean on-topic |
| :--- | ---: |
| quota 0.5, two channels | 59% |
| RRF, two channels | 64% |
| **RRF, three channels** | **84%** |
| quota 0.5 + lexical folded into the seeded side | 59% |

**The last row is the finding.** Adding the lexical channel to the quota
changes nothing at all. The quota interleaves traversal against one "seeded"
list, so appending lexical results to that list puts them behind entries that
the 25-relation budget already truncates — they never surface. The channel is
only reachable when the merge treats it as a peer.

So neither change carries the gain alone: fusion alone is +5, and the third
channel is worth +20 *only* under fusion. That is an argument about
architecture rather than tuning — a fixed-share merge cannot accommodate a new
generator without re-tuning the share, while RRF admits one for free.

The FAA question is the exception, gaining little (52% to 56%). It names a
well-connected entity that traversal already reaches, which is the case the
earlier quota measurement also found least improvable.

A caveat this measurement inherits: on-topic share counts keyword overlap,
which flatters lexical matching by construction. The direction is large enough
to survive that bias, but the magnitude should not be quoted without the blind
answer comparison that settled the quota question.
