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
python evaluation/index_corpus.py --max-chunks 10 \
    --model MiniMax-M2.7 \
    --fallback-models gemma-4-31B-it DeepSeek-V3.2 gpt-oss-120b DeepSeek-V3.1
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

Providers that exhaust credits mid-run are the normal case on shared routers;
`--fallback-models` and `--max-retries` let a long run ride through them. List the
fallbacks in descending quality order, so degradation is graceful — the chain is
tried left to right and the first model with credit wins.

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
