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
    --model DeepSeek-V3.2 \
    --fallback-models Meta-Llama-3.3-70B-Instruct MiniMax-M2.7
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
`--fallback-models` and `--max-retries` let a long run ride through them.

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
`text-embedding-3-small`, Llama-3.3-70B with fallbacks.

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
