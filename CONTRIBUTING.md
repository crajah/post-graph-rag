# Contributing to post-graph-rag

Bug reports, failing test cases and pull requests are all welcome. This file
covers the parts of the setup that are specific to this project — the ones you
would otherwise discover by losing an afternoon to them.

## What you need

**PostgreSQL with pgvector.** The test suite talks to a real database rather
than mocking one, because most of what it verifies — generated columns, JSONB
predicates, vector search, cascade behaviour — lives in the database and cannot
be mocked without testing the mock instead.

```bash
# macOS
brew install postgresql@16 pgvector

# Debian/Ubuntu (match your server version)
sudo apt install postgresql-16 postgresql-16-pgvector

createdb post_graph_test
psql -d post_graph_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**No LLM credentials.** Every test uses an in-process fake, so the suite is
free, offline and deterministic. A test that reaches the network is a bug in
the test.

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

export POSTGRES_TEST_URI=postgresql://localhost:5432/post_graph_test
pytest -q                  # ~500 tests, around three minutes
pytest tests/test_engine.py -q          # one file
pytest -q -k supersession               # one behaviour
```

The suite must be green before you open a pull request. CI runs the same
command on Python 3.9 and 3.13 against a pgvector container.

## Linting

```bash
pip install ruff
ruff check .               # must pass; CI enforces it
ruff check . --fix         # fixes most of what it finds
```

The configuration in `pyproject.toml` selects `F`, `B`, `E9` and `I` — rules
that catch defects, not house style. **Do not run `ruff format`.** It would
rewrite several thousand lines and take the history behind them with it, for
no defect caught. Formatting is deliberately not enforced; match the style of
the file you are editing.

## The evaluation harnesses

Everything under `evaluation/` is research tooling rather than library code,
and it is not covered by the test suite.

- It needs an OpenAI-compatible endpoint and **costs real money to run**. A
  full LongMemEval sweep is thousands of LLM calls over several hours.
- Datasets are fetched, not committed: `evaluation/longmemeval/fetch.sh` and
  `evaluation/ectqa/fetch.sh`.
- Results files behind published claims **are** committed, by name. They are
  allowlisted in `.gitignore`, so a new one needs an explicit `!` entry —
  otherwise the dataset globs will swallow it silently.

You do not need any of this to contribute to the library.

## Pull requests

- **One change per pull request.** A bug fix and a refactor in the same diff
  take several times longer to review than the two separately.
- **A test that fails before your change and passes after** is the most useful
  thing in a pull request. For a bug, that test is more valuable than the fix;
  send it on its own if you would rather.
- **Explain the why in the commit message.** What changed is visible in the
  diff. Why it changed, and what you ruled out, is not.
- **Say what you measured.** For anything touching retrieval or extraction
  quality, a claim of improvement needs a number. Re-indexing alone moves this
  benchmark across a 15-point band, so an A/B that rebuilds the graph per arm
  is measuring the extraction model rather than your change. Hold one graph
  fixed across arms — `evaluation/longmemeval/ablate_retrieval.py` shows the
  pattern.

## Things worth knowing before you change them

- **The write path fails loudly on purpose.** Extraction that produces nothing
  usable raises rather than writing placeholder structure, because a bad edge
  is indistinguishable from a real one once stored. Please do not add a
  fallback that writes something.
- **Retrieval telemetry is the one exception**, and it is documented as such:
  read-side bookkeeping must never fail the query it describes.
- **Supersession depends on document order**, not insertion time. Backfilling
  older documents after newer ones must not retire the newer facts.

## Reporting a bug

Include the failing question or document, the config you used, and what you
expected. If it involves retrieval quality, `rag.query_data(...)` returns the
retrieved context without synthesis, which usually shows whether the problem
is retrieval or the model.

## Licence

Apache 2.0. Contributions are accepted under the same licence.
