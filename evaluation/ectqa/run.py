"""ECT-QA: time-sensitive QA over earnings call transcripts.

From "RAG Meets Temporal Graphs" (arXiv:2510.13590). 480 transcripts, 24
companies, 2020-2024, with questions typed by temporal shape — single-time,
multi-time and relative-time — plus a deliberate `unanswerable` class.

This is a harder register than LongMemEval and a closer match to what this
library is for. A company restates the same metrics every quarter for four
years, so "free cash flow" has sixteen values and the question names which one
it wants. Retrieval that ignores time returns whichever quarter embeds best.

One realm, one space per company. That is the axis these questions want: a
single-company question filters to its space, and a cross-company question
queries `__all__`, which post-graph 0.8.0 honours by dropping the space filter
and searching every company in the realm. Realm-per-company would have made the
cross-company questions unanswerable, since realms are physically separate
schemas and vector search does not span them.

Every transcript states its quarter in the indexed text, which is the point:
the graph learns when each statement was made without inferring it from prose.
"""
import argparse
import asyncio
import collections
import importlib.util
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

# Loaded by path, not by name. This file is also called run.py, and its own
# directory is on sys.path ahead of everything at startup, so `import run` is a
# coin toss decided by insert order — it would quietly import itself the moment
# the paths above are reordered.
_spec = importlib.util.spec_from_file_location(
    "longmemeval_run", HERE.parent / "longmemeval" / "run.py")
assert _spec and _spec.loader
_lme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lme)
judge_panel = _lme.judge_panel  # the LongMemEval panel, reused unchanged

from post_graph import RESERVED_SPACE_ALL  # noqa: E402
from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402

# A quarter is a period, but a relation needs a date. The quarter end is the
# defensible choice: the call reports on the quarter that just closed, so the
# statements are true as of its end and are asserted shortly after.
QUARTER_END = {"q1": "-03-31", "q2": "-06-30", "q3": "-09-30", "q4": "-12-31"}

# Finance predicates. Free-text predicates fragment retrieval on this register —
# the LightRAG comparison in evaluation/README.md measured a 46% distinct-label
# ratio on 10-Ks against 11% on biography — and these questions turn on
# comparing the same measure across quarters, which needs one label for it.
FINANCE_PREDICATES = [
    "reported_revenue", "reported_earnings", "reported_margin", "reported_growth",
    "reported_cash_flow", "reported_guidance", "increased", "decreased",
    "acquired", "divested", "launched", "discontinued", "invested_in",
    "expanded_into", "faced_headwind", "benefited_from", "returned_capital",
    "repurchased", "employs", "led_by", "partnered_with", "supplies",
]

ANSWER_INSTRUCTION = (
    "You are answering a question about earnings call transcripts. The facts "
    "below carry the quarter they were stated in. Match the quarter the question "
    "asks about — a value from the wrong quarter is a wrong answer. If the facts "
    "do not support an answer, say exactly: unanswerable."
)


def transcript_files(data_dir: pathlib.Path):
    """Group the corpus by company. Filenames are sector-CODE-year-quarter.json."""
    by_company = collections.defaultdict(list)
    for path in sorted(data_dir.glob("*.json")):
        parts = path.stem.split("-")
        if len(parts) != 4:
            continue
        _sector, code, year, quarter = parts
        by_company[code].append((year, quarter, path))
    for code in by_company:
        by_company[code].sort(key=lambda t: (t[0], t[1]))
    return by_company


def question_companies(q):
    """Stock codes the question's evidence spans. Empty for unanswerable ones."""
    return {e["stock_code"] for e in q.get("evidence_list", [])}


async def index_company(rag, quarters, space):
    """Index a company's transcripts oldest first, each stamped with its quarter.

    Document order matters: supersession resolves by it, so a later quarter must
    be indexed after an earlier one for a restated figure to close the earlier
    assertion rather than sit beside it.
    """
    for year, quarter, path in quarters:
        doc = json.loads(path.read_text())
        text = doc.get("cleaned_content") or doc.get("raw_content") or ""
        if not text.strip():
            continue
        as_of = f"{year}{QUARTER_END[quarter]}"
        # The date is stated in the text as well as in the metadata. Extraction
        # reads prose, not the metadata dict — a lesson from LongMemEval, where
        # dates living only in DocumentMetadata.extra never reached the prompt
        # and every temporal question failed.
        header = (f"[{doc.get('company_name', path.stem)} earnings call, "
                  f"{year} {quarter.upper()}, quarter ending {as_of}]\n\n")
        await rag.index_text(header + text, metadata=DocumentMetadata(
            document=f"{doc.get('stock_code', '')}-{year}-{quarter}",
            source="ect", space=space, extra={"quarter_end": as_of},
        ))
    return len(quarters)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--questions", default=str(HERE / "local_questions_old.json"))
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--judges", nargs="*",
                    default=["MiniMax-M2.7", "gpt-oss-120b", "DeepSeek-V3.2"])
    # The router puts models into hour-long cooldowns and exhausts provider
    # credits mid-run. Fallbacks are what let a multi-hour indexing job survive
    # that instead of dying an hour in with nothing written.
    ap.add_argument("--fallback-models", nargs="*", default=[
        "gemini-3.5-flash-lite", "google/gemma-4-26b-a4b-it-maas",
        "Meta-Llama-3.3-70B-Instruct"])
    ap.add_argument("--realm", default="ectqa")
    ap.add_argument("--companies", type=int, default=6,
                    help="companies to index; each gets a space")
    ap.add_argument("--per-company", type=int, default=10)
    ap.add_argument("--cross-company", type=int, default=10,
                    help="questions spanning several companies, asked against __all__")
    ap.add_argument("--unanswerable", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-concurrency", type=int, default=4)
    ap.add_argument("--query-concurrency", type=int, default=6)
    ap.add_argument("--chunk-concurrency", type=int, default=8)
    # Attempts per model before failing over. Deliberately small: the router's
    # 429s carry "try again in 3600 seconds", so retrying the same model is
    # waiting an hour by instalments when a working model is one call away.
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--skip-index", action="store_true",
                    help="reuse an already-indexed realm and only run the questions")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    questions = json.loads(pathlib.Path(args.questions).read_text())
    corpus = transcript_files(pathlib.Path(args.data))

    def corpus_key(code):
        for candidate in (code, code.replace(" ", "_"), code.split()[0]):
            if candidate in corpus:
                return candidate
        return None

    single, cross, unanswerable = collections.defaultdict(list), [], []
    for q in questions:
        if q["answer"] == "unanswerable":
            unanswerable.append(q)
            continue
        codes = question_companies(q)
        if len(codes) == 1:
            single[next(iter(codes))].append(q)
        elif codes:
            cross.append((codes, q))

    # Rank by usefulness to *both* question kinds. Cross-company questions span
    # three to five companies each, so choosing purely by single-company volume
    # picks a set that covers almost no cross-company question — the top four by
    # that measure cover 10 of 87, against 49 for the top six counted this way.
    cross_freq = collections.Counter(c for codes, _ in cross for c in codes)
    ranked = sorted(((c, qs) for c, qs in single.items() if corpus_key(c)),
                    key=lambda kv: -(len(kv[1]) + 3 * cross_freq.get(kv[0], 0)))[:args.companies]
    chosen = {c for c, _ in ranked}
    rng = random.Random(args.seed)

    # A space per company, all in one realm. `space` is the logical axis, so a
    # question about one company filters to its space and a question spanning
    # several searches every space at once.
    spaces = {code: code.replace(" ", "_").lower() for code, _ in ranked}

    selected = []
    for code, qs in ranked:
        picked = qs[:]
        rng.shuffle(picked)
        selected += [(spaces[code], code, q) for q in picked[:args.per_company]]

    # Only cross-company questions whose companies are all indexed. One missing
    # company makes the question unanswerable for a reason that has nothing to
    # do with retrieval.
    eligible = [q for codes, q in cross if codes <= chosen]
    rng.shuffle(eligible)
    selected += [(RESERVED_SPACE_ALL, "+".join(sorted(question_companies(q))), q)
                 for q in eligible[:args.cross_company]]

    pool = unanswerable[:]
    rng.shuffle(pool)
    selected += [(RESERVED_SPACE_ALL, "-", q) for q in pool[:args.unanswerable]]

    print(f"realm={args.realm} | {len(ranked)} companies as spaces | "
          f"{len(selected)} questions | model={args.model}")
    for code, qs in ranked:
        print(f"  {code:<10} space={spaces[code]:<10} "
              f"{len(corpus[corpus_key(code)]):>3} quarters, {len(qs):>3} questions")
    print(f"  cross-company eligible: {len(eligible)} (asking {min(len(eligible), args.cross_company)})")
    print(f"judges: {', '.join(args.judges)}\n")

    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=args.max_retries,
                                          fallback_models=args.fallback_models,
                                          retry_deadline_secs=900))
                  for n in args.judges}

    rag = GraphRAG(RAGConfig(
        model=args.model, realm=args.realm, schema_per_realm=True,
        embedding_model=args.embedding_model, embedding_dim=1536,
        embed_relations=True, merge_strategy="rrf",
        predicate_vocabulary=FINANCE_PREDICATES, extract_validity=True,
        max_concurrent_chunks=args.chunk_concurrency,
        fallback_models=args.fallback_models,
        max_retries=args.max_retries, retry_deadline_secs=1800))
    await rag.initialize()

    results, by_type, degraded = [], collections.defaultdict(list), []
    started = time.time()

    try:
        if not args.skip_index:
            isem = asyncio.Semaphore(args.index_concurrency)

            async def index_one(code):
                async with isem:
                    t0 = time.time()
                    try:
                        n = await index_company(rag, corpus[corpus_key(code)], spaces[code])
                        print(f"[{code}] indexed {n} quarters into space "
                              f"'{spaces[code]}' in {time.time()-t0:.0f}s", flush=True)
                    except Exception as e:
                        degraded.append({"company": code,
                                         "reason": f"{type(e).__name__}: {e}"})
                        print(f"[{code}] INDEX FAILED {type(e).__name__}: "
                              f"{str(e)[:70]}", flush=True)

            await asyncio.gather(*(index_one(c) for c, _ in ranked))
            print(f"indexing done in {time.time()-started:.0f}s\n", flush=True)

        qsem = asyncio.Semaphore(args.query_concurrency)

        async def ask(space, label, q):
            async with qsem:
                t0 = time.time()
                try:
                    out = await rag.query(
                        f"{ANSWER_INSTRUCTION}\n\nQuestion: {q['question']}",
                        param=QueryParam(mode="mix", top_k=args.top_k, space=space))
                    answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()
                except Exception as e:
                    degraded.append({"question": q["question"][:80],
                                     "reason": f"{type(e).__name__}: {e}"})
                    print(f"  QUERY FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
                    return
                t_query = time.time() - t0

                gold = q["answer"]
                if gold == "unanswerable":
                    # Whether the system declined, judged by rule. A panel would
                    # argue about the wording of a refusal rather than the fact
                    # of it.
                    ok = ("unanswerable" in answer.lower()
                          or "not support" in answer.lower()
                          or "do not contain" in answer.lower())
                    votes = {"rule": ok}
                    kind = "unanswerable"
                else:
                    ok, votes = await judge_panel(judge_llms, q["question"], gold, answer)
                    kind = ("cross-company" if space == RESERVED_SPACE_ALL
                            else q["question_type"].split("|")[0])

                by_type[kind].append(bool(ok))
                results.append({
                    "space": space, "companies": label, "type": kind,
                    "question": q["question"], "gold": gold,
                    "num_hops": q.get("num_hops"),
                    "reasoning_type": q.get("reasoning_type"),
                    "correct": bool(ok), "votes": votes,
                    "query_secs": round(t_query, 1), "answer": answer[:800]})
                print(f"  {kind:<18} {'CORRECT' if ok else 'incorrect':<9} "
                      f"{t_query:>5.1f}s  [{len(results)}/{len(selected)}]", flush=True)

        await asyncio.gather(*(ask(s, lab, q) for s, lab, q in selected))
    finally:
        await rag.close()

    print("\n" + "-" * 60)
    total = sum(1 for r in results if r["correct"])
    for kind, marks in sorted(by_type.items()):
        print(f"  {kind:<20} {sum(marks):>3}/{len(marks):<3} "
              f"{100*sum(marks)/max(1,len(marks)):>5.0f}%")
    print(f"  {'OVERALL':<20} {total:>3}/{len(results):<3} "
          f"{100*total/max(1,len(results)):>5.0f}%")
    if degraded:
        print(f"\n  !! {len(degraded)} failures — not reportable")

    pathlib.Path(args.out).write_text(json.dumps(
        {"benchmark": "ECT-QA", "model": args.model, "judges": args.judges,
         # Which models actually served, not merely which were configured. A
         # router cooldown mid-run silently splits extraction across the
         # fallback chain, and a graph built that way is not attributable to
         # any one model.
         "served": dict(rag.llm.served),
         "judges_served": {n: dict(j.served) for n, j in judge_llms.items()},
         "realm": args.realm, "spaces": spaces, "n": len(results),
         "accuracy": total / max(1, len(results)),
         "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded, "results": results}, indent=2))
    print("\n  models that served extraction and answering:")
    for name, n in rag.llm.served.most_common():
        print(f"    {name:<34} {n:>6}")
    print(f"  wrote {args.out}  ({time.time()-started:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
