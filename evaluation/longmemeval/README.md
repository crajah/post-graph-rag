# LongMemEval

An external benchmark, added because every other measurement in this repository
is structural — graph density, label ratios, on-topic share of retrieved
relations — and none of it says whether the system answers questions correctly
against someone else's data and someone else's scoring.

It is also the benchmark Zep publish on ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)),
which makes it the fairest available comparison for a temporally-aware graph.

```bash
./fetch.sh                       # oracle variant, ~15MB, public and ungated
python -u run.py --limit 20 --types temporal-reasoning knowledge-update \
    --concurrency 10 --chunk-concurrency 8
```

## First reportable result

`google/gemma-4-26b-a4b-it-maas`, `gemini-embedding-001`, judged by
`gemini-3.6-flash`, RRF merge, oracle variant, 20 instances:

| | accuracy |
| :--- | ---: |
| temporal-reasoning | 4/16 — 25% |
| knowledge-update | 1/4 — 25% |
| **overall** | **5/20 — 25%** |

Median query latency 3.9s; median indexing 129s per instance (max 705s).
Zero degraded instances, zero provider errors — the first run of five where
that was true.

**This is a poor result and the cause is not what the benchmark is testing.**

Of the 15 wrong answers, **all 15 are retrieval failures**. Eight decline
explicitly; the other seven answer while saying "there is no information
regarding…". None finds the evidence and reasons badly. The misses are not
even temporal:

| question | gold answer |
| :--- | :--- |
| where do you keep your old sneakers | in a shoe rack in my closet |
| what time do you wake up | 6:45 AM |
| what is your current status | Premier Silver |

These are single facts stated plainly in one session. A diagnostic on one
instance shows why: of 32 relations extracted from a conversation, **none
carried `valid_from`**, and the relations themselves were abstractions —
`Seasonal flavors -[creates]-> Product variety` — rather than the concrete
personal facts the questions ask about.

So the number measures **the extraction prompt on conversational text**. It was
tuned on filings and encyclopedic prose, where the significant entities are
named things and validity is stated in the sentence. Chat encodes its facts
differently: preferences, locations, times and possessions, with the date in the
envelope rather than the prose. The graph, the temporal model and the retrieval
fusion are barely exercised, because what reaches them is already the wrong
material.

## Caveats

- **Oracle variant**, which contains only the evidence sessions. That is easier
  than the S and M haystacks Zep report on — being behind on the easier variant
  understates the gap rather than flattering it.
- **20 instances**, one seed, one model. Indicative, not a benchmark result.
- Scored by an LLM judge, as the benchmark itself specifies, with the judge held
  distinct from the answering model.

## Design decisions in the harness

These determine whether the number means anything, so they are recorded rather
than left implicit.

- **One document per session**, turns rendered with roles intact — who said a
  thing is often the fact being asked about.
- **Sessions indexed in date order.** Supersession resolves contradictions by
  document order, so indexing a later session first would invert which fact
  wins. That is the difference between right and wrong on `knowledge-update`.
- **Session date written into the document body**, not metadata. An early run
  scored 0/3 because the date lived in `DocumentMetadata.extra`, which never
  reaches the prompt — and for "how many weeks between X and Y", the date is not
  context for the answer, it is the answer.
- **The question's own date is supplied.** "How many weeks since" is measured
  from it and the benchmark gives it to every system, so withholding it would be
  handicapping rather than rigour.
- **A realm per instance**, dropped afterwards. Instances are independent
  haystacks; a shared graph would let one answer another.
- **Asked through `query()`**, the shipped path. A reconstruction of retrieval
  once understated a baseline by twenty points here.

## Fail-closed, and why

An instance whose sessions do not all index raises `DegradedRun`, is skipped,
counted, and **excluded from the denominator**, and the summary refuses to
present the accuracy as reportable while that count is non-zero.

This exists because an early run reported 0/3 while the model router was
returning 402s: extraction was failing, the graph was partial, and the engine
answered "I cannot determine" for questions whose evidence had never been
indexed — indistinguishable, in the result table, from a reasoning failure. A
later run skipped 11 of 20 and said so. A benchmark that averages provider
outages into the score measures the provider.

`run.py` also preflights the model — one real session indexed, relations
confirmed — before starting a sweep. Four runs died on model availability: a
name the router advertised but did not serve, and a name differing from the
router's by a single character. Each cost an hour to discover what the preflight
costs seconds.

## What to fix first

Extraction, not retrieval. A conversational extraction prompt that keeps
personal facts, and that attaches the session date to the relations it produces,
would give the temporal model something to reason over. Until then this number
is a measurement of prompt fit, and re-running it against a different merge
strategy or hop budget would be measuring noise.
