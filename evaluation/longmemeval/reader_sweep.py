"""Reader sweep: one graph per instance, several answering models over it.

`ablate_retrieval.py` holds the graph fixed and varies retrieval flags. This
holds the graph fixed and varies the model that reads it, which is the other
half of the same question: when a benchmark score stops moving, is the limit in
what was extracted or in what reads it?

The pairing is what makes it answerable. Each instance is indexed exactly once,
by `--index-model`, and every arm then answers from that byte-identical graph.
Indexing variance -- which this repository has measured at wider than any effect
worth testing -- is absent by construction, so a difference between arms is
attributable to the reader.

Swapping the arm works because `LLMService` reads `config.model` at call time
and `GraphRAG` holds the same config object, so assigning to `rag.config.model`
after indexing changes the answering model without touching the graph.

Scoring is layered rather than judged. On ECT-QA a judge panel scored six points
below a deterministic metric over identical answers, so every answer here is
scored four ways and all four are stored:

    numeric F1   when the gold states figures
    period F1    when the gold is a period
    cosine       otherwise -- gold is terse, answers are prose
    judge panel  always, as the comparison rather than the authority

The headline is the layered cascade; the panel is reported beside it so the two
metrics can be compared on the same answers.
"""
import argparse
import asyncio
import importlib.util
import json
import pathlib
import random
import sys
import time
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from run import (  # noqa: E402
    CONVERSATIONAL_PROMPT, DegradedRun, index_instance, judge_panel, parse_date,
)

_spec = importlib.util.spec_from_file_location(
    "_ectqa_run", HERE.parents[0] / "ectqa" / "run.py")
_ect = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_ect)
except SystemExit:
    pass
numeric_f1, period_f1 = _ect.numeric_f1, _ect.period_f1
answer_similarity = _ect.answer_similarity

from post_graph_rag import GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService                   # noqa: E402


def family(model: str) -> str:
    """Coarse vendor/lineage key, so a sibling cannot grade its own family.

    gemini-3.6 and gemini-3.7 share a lineage: letting one grade the other is
    not self-judging but is the next thing to it, and it would apply to only
    one arm of the pair, which is worse than applying to neither.
    """
    m = model.split("/")[-1].lower()
    for key in ("gemini", "gemma", "minimax", "gpt-oss", "deepseek", "llama"):
        if key in m:
            return key
    return m


def panel_for(arm: str, pool: list) -> list:
    """The three pool members that are not the arm's own family."""
    panel = [p for p in pool if family(p) != family(arm)]
    if len(panel) != 3:
        raise SystemExit(
            f"panel for {arm!r} has {len(panel)} judges, not 3: {panel}. "
            f"The pool must contain exactly one model of the arm's family.")
    return panel


async def layered_score(llm, gold, answer, f1_threshold, sim_threshold):
    """Deterministic where the gold allows it, embedding similarity otherwise.

    `gold` is coerced because 32 of LongMemEval's 500 answers are bare integers
    rather than strings, and those are exactly the ones numeric F1 exists to
    score -- so an uncoerced gold fails on the subset that matters most.
    """
    gold, answer = str(gold), str(answer)
    f1 = numeric_f1(gold, answer)
    if f1 is not None:
        return ("numeric_f1", f1, f1 >= f1_threshold)
    pf1 = period_f1(gold, answer)
    if pf1 is not None:
        return ("period_f1", pf1, pf1 >= f1_threshold)
    sim = await answer_similarity(llm, gold, answer)
    return ("cosine", sim, sim >= sim_threshold)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "oracle.json"))
    ap.add_argument("--index-model", default="gemini-3.7-flash",
                    help="builds the graph once per instance; never varied")
    ap.add_argument("--arms", nargs="*", default=[
        "gemini-3.7-flash", "gemini-3.6-flash",
        "google/gemma-4-26b-a4b-it-maas", "gpt-oss-120b"])
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    # A pool of four, from which each arm is judged by the three that are not
    # its own family. Three keeps the vote odd, so no ballot ties.
    ap.add_argument("--judge-pool", nargs="*", default=[
        "gemini-3.7-flash", "google/gemma-4-26b-a4b-it-maas",
        "MiniMax-M2.7", "gpt-oss-120b"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    # One pool per in-flight instance. The default of 10/10 means concurrency 20
    # asks PostgreSQL for 200 connections at once, and since it forks a backend
    # per connection, pool creation times out before any work starts. Each
    # instance is one small graph, so a handful of connections is ample.
    ap.add_argument("--pool-min", type=int, default=1)
    ap.add_argument("--pool-max", type=int, default=4)
    ap.add_argument("--chunk-concurrency", type=int, default=8)
    ap.add_argument("--f1-threshold", type=float, default=0.60)
    ap.add_argument("--similarity-threshold", type=float, default=0.62)
    ap.add_argument("--out", default=str(HERE / "reader_sweep.json"))
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text())
    if args.types:
        data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900))
                  for n in args.judge_pool}
    panels = {arm: panel_for(arm, args.judge_pool) for arm in args.arms}
    score_llm = LLMService(RAGConfig(embedding_model=args.embedding_model))

    print(f"{len(data)} instances | one graph each, built by {args.index_model}")
    print(f"arms: {', '.join(args.arms)} | repeats: {args.repeats}")
    for arm in args.arms:
        print(f"  {arm:<32} judged by {', '.join(panels[arm])}")
    print(flush=True)

    layered, judged = defaultdict(list), defaultdict(list)
    per_instance, degraded = [], []
    sem = asyncio.Semaphore(args.concurrency)
    # Pool construction is the burst that overwhelms the server, not steady-state
    # querying. Serialising just that keeps query concurrency at full width.
    init_lock = asyncio.Lock()
    started = time.time()

    async def one(i, inst):
        async with sem:
            rag = GraphRAG(RAGConfig(
                model=args.index_model, realm=f"rsweep_{args.seed}_{i}",
                schema_per_realm=True, embedding_model=args.embedding_model,
                embedding_dim=1536, embed_relations=True, merge_strategy="rrf",
                max_concurrent_chunks=args.chunk_concurrency,
                pool_min_size=args.pool_min, pool_max_size=args.pool_max,
                extraction_prompt=CONVERSATIONAL_PROMPT,
                max_retries=40, retry_deadline_secs=1800))
            async with init_lock:
                await rag.initialize()
            try:
                try:
                    await index_instance(rag, inst)
                except DegradedRun as e:
                    degraded.append({"question_id": inst["question_id"], "reason": str(e)})
                    print(f"{inst['question_id']:<16} SKIPPED ({str(e)[:44]})", flush=True)
                    return

                asked = (f"Today is {parse_date(inst.get('question_date',''))}. "
                         f"{inst['question']}") if inst.get("question_date") else inst["question"]
                gold = str(inst["answer"])
                row = {"question_id": inst["question_id"], "type": inst["question_type"],
                       "gold": gold, "arms": {}}

                for arm in args.arms:
                    # The graph is already built; this only changes who reads it.
                    rag.config.model = arm
                    lmarks, jmarks, detail = [], [], []
                    for _ in range(args.repeats):
                        out = await rag.query(asked, param=QueryParam(mode="mix", top_k=8))
                        answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()
                        metric, value, ok = await layered_score(
                            score_llm, gold, answer,
                            args.f1_threshold, args.similarity_threshold)
                        arm_judges = {n: judge_llms[n] for n in panels[arm]}
                        jok, votes = await judge_panel(arm_judges, inst["question"],
                                                       gold, answer)
                        lmarks.append(bool(ok)); jmarks.append(bool(jok))
                        detail.append({"answer": answer[:400], "metric": metric,
                                       "value": value, "layered": bool(ok),
                                       "judge": bool(jok), "votes": votes})
                    ls, js = sum(lmarks)/len(lmarks), sum(jmarks)/len(jmarks)
                    layered[arm].append(ls); judged[arm].append(js)
                    row["arms"][arm] = {"layered": ls, "judge": js, "runs": detail}

                per_instance.append(row)
                print(f"{inst['question_id']:<16} " + "  ".join(
                    f"{a.split('/')[-1][:14]}={row['arms'][a]['layered']:.2f}"
                    for a in args.arms), flush=True)
            except Exception as e:
                degraded.append({"question_id": inst["question_id"],
                                 "reason": f"{type(e).__name__}: {e}"})
                print(f"{inst['question_id']:<16} FAILED {type(e).__name__}: "
                      f"{str(e)[:50]}", flush=True)
            finally:
                await rag.close()

    await asyncio.gather(*(one(i, inst) for i, inst in enumerate(data)))

    print("\n" + "-" * 68)
    print(f"  {'arm':<32} {'layered':>8} {'judge':>8}   n")
    for arm in args.arms:
        if not layered[arm]:
            continue
        lm = sum(layered[arm]) / len(layered[arm])
        jm = sum(judged[arm]) / len(judged[arm])
        print(f"  {arm:<32} {lm*100:7.1f}% {jm*100:7.1f}%  {len(layered[arm])}")
    if degraded:
        print(f"\n  !! {len(degraded)} of {len(data)} degraded — not reportable")
    print(f"  {time.time()-started:.0f}s total")

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "LongMemEval", "index_model": args.index_model,
        "arms": args.arms, "judge_pool": args.judge_pool,
        "panels": panels, "repeats": args.repeats,
        "f1_threshold": args.f1_threshold,
        "similarity_threshold": args.similarity_threshold,
        "means_layered": {a: (sum(v)/len(v) if v else None) for a, v in layered.items()},
        "means_judge": {a: (sum(v)/len(v) if v else None) for a, v in judged.items()},
        "degraded": degraded, "degraded_count": len(degraded),
        "reportable": not degraded, "instances": per_instance}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
