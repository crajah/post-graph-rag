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

## Result

`gemini-3.6-flash` extracting and answering, `gemini-embedding-001`, RRF merge,
conversational extraction prompt, oracle variant, 20 instances. Judged by a
panel of **MiniMax-M2.7, gpt-oss-120b and DeepSeek-V3.2**, majority vote:

| | accuracy |
| :--- | ---: |
| **temporal-reasoning** | **13/16 — 81%** |
| knowledge-update | 2/4 — 50% |
| **overall** | **15/20 — 75%** |

Zero degraded instances. Median query latency 7.9s, median indexing 50s per
instance, 570s for the run.

**The judges agreed almost completely** — MiniMax 95%, gpt-oss 95%, DeepSeek
100% agreement with the majority verdict. One judgement was contested across
sixty votes, so the figure is a property of the answers rather than of who
marked them. That mattered to establish: re-grading a fixed set of stored
answers with a different single judge had previously moved a score by five
points.

### How it got here

| | overall | what changed |
| :--- | ---: | :--- |
| gemma-4-26b, document extraction prompt | 25% | baseline |
| MiniMax-M2.7, conversational prompt | 45% | **the extraction prompt** |
| gemini-3.6-flash, single judge | 70% | **the answering model** |
| gemini-3.6-flash, three-judge panel | **75%** | judging scheme |

Two interventions carried the gain, and they were worth about the same.

**The extraction prompt was the first.** The library's default forbids exactly
what a chat log is made of: never emit a pronoun as an entity, never emit a
possessive role phrase, emit only the stable entity nameable in a *different*
document. In a conversation the central entity is the speaker — "I", "my" — and
the facts are possessive: my sneakers, my wake-up time, my loyalty tier. Rules
that are right for filings discard the entire subject matter of a chat. Before
the fix, a diagnostic on one instance found 32 extracted relations, **none**
carrying `valid_from`, and the relations themselves were abstractions like
`Seasonal flavors -[creates]-> Product variety`. The conversational prompt
inverts those rules and makes the session date mandatory on every triple.

**The answering model was the second**, holding prompt and merge fixed. That is
consistent with everything else measured in this repository: model choice moves
results at least as much as design does.

### What still fails

Five wrong answers, and they are no longer retrieval failures of the original
kind. Two produce a specific wrong value (`3 tops` where the gold is `five`),
two decline on questions whose evidence is present, and one answers a
possession question with the wrong location. The date-arithmetic cluster that
dominated earlier runs is largely gone — 81% on temporal-reasoning is the
category this design exists for, and it was 25% before the extraction fix.

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
