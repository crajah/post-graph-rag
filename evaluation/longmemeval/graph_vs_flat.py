"""Rusu et al.'s Experiment 1, run inside post-graph-rag.

arXiv:2608.28978 report their knowledge graph LOSING to a flat vector baseline
on LongMemEval (LLM-judge 0.454 vs 0.536), concluding structure does not pay.
The claim is testable directly here, and more cleanly than they ran it: our
engine ships both retrieval modes in one codebase, so the same graph, the same
model, the same judges and the same questions differ only in `mode` --
`naive` is flat top-k vector RAG over the chunks, `mix` is the full graph
path. Their comparison used two separate systems; ours isolates the mechanism.

One graph is built per instance (belief-time grounding on, as in the headline
runs) and answered twice, once per mode. Judged by the same family-excluded
panel as the reader sweep. --retry-from and per-arm tolerance are inherited by
reusing the sweep's machinery for indexing and judging.
"""
import argparse
import asyncio
import json
import pathlib
import random
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from reader_sweep import panel_for  # noqa: E402
from run import (  # noqa: E402
    CONVERSATIONAL_PROMPT,
    DegradedRun,
    index_instance,
    judge_panel,
    parse_date,
)

from post_graph_rag import GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402

MODES = ("naive", "mix")   # their Baseline RAG, our Graph RAG


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "oracle.json"))
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--judge-pool", nargs="*", default=[
        "google/gemma-4-26b-a4b-it-maas", "Meta-Llama-3.3-70B-Instruct", "gpt-oss-120b"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--pool-min", type=int, default=1)
    ap.add_argument("--pool-max", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "graph_vs_flat.json"))
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text())
    if args.types:
        data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    # the model answers both modes, so it is excluded from its own panel
    panel = panel_for(args.model, args.judge_pool)
    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900)) for n in panel}
    print(f"{len(data)} instances | model={args.model} | modes: {', '.join(MODES)}")
    print(f"judges: {', '.join(panel)}\n", flush=True)

    by_type = defaultdict(lambda: defaultdict(list))
    per_instance, degraded = [], []
    sem = asyncio.Semaphore(args.concurrency)
    init_lock = asyncio.Lock()

    async def one(i, inst):
        async with sem:
            rag = GraphRAG(RAGConfig(
                model=args.model, realm=f"gvf_{args.seed}_{i}", schema_per_realm=True,
                embedding_model=args.embedding_model, embedding_dim=1536,
                embed_relations=True, merge_strategy="rrf",
                extraction_prompt=CONVERSATIONAL_PROMPT,
                pool_min_size=args.pool_min, pool_max_size=args.pool_max,
                max_retries=40, retry_deadline_secs=600))
            async with init_lock:
                await rag.initialize()
            try:
                try:
                    await index_instance(rag, inst)
                except DegradedRun as e:
                    degraded.append({"question_id": inst["question_id"], "reason": str(e)})
                    return
                asked = (f"Today is {parse_date(inst.get('question_date',''))}. "
                         f"{inst['question']}") if inst.get("question_date") else inst["question"]
                row = {"question_id": inst["question_id"], "type": inst["question_type"], "modes": {}}
                for mode in MODES:
                    out = await rag.query(asked, param=QueryParam(mode=mode, top_k=8))
                    ans = (out["answer"] if isinstance(out, dict) else str(out)).strip()
                    ok, _v = await judge_panel(judge_llms, inst["question"],
                                               inst["answer"], ans)
                    row["modes"][mode] = bool(ok)
                    by_type[inst["question_type"]][mode].append(1 if ok else 0)
                per_instance.append(row)
                print(f"{inst['question_id']:<16} " +
                      "  ".join(f"{m}={'Y' if row['modes'][m] else 'n'}" for m in MODES),
                      flush=True)
            except Exception as e:
                degraded.append({"question_id": inst["question_id"],
                                 "reason": f"{type(e).__name__}: {e}"})
            finally:
                await rag.close()

    await asyncio.gather(*(one(i, inst) for i, inst in enumerate(data)))

    print("\n" + "=" * 60)
    ORDER = ["single-session-user", "single-session-assistant", "knowledge-update",
             "multi-session", "temporal-reasoning", "single-session-preference"]
    tot = {m: [] for m in MODES}
    print(f"  {'type':<28}" + "".join(f"{m:>10}" for m in MODES))
    for t in ORDER:
        if t not in by_type:
            continue
        line = f"  {t:<28}"
        for m in MODES:
            v = by_type[t][m]
            line += f"{(sum(v)/len(v) if v else 0):>10.3f}"
            tot[m] += v
        print(line)
    print(f"  {'OVERALL':<28}" + "".join(
        f"{(sum(tot[m])/len(tot[m]) if tot[m] else 0):>10.3f}" for m in MODES))
    print("\n  their result: flat 0.536 judge vs graph 0.454 -- structure lost.")
    print(f"  {len(per_instance)} scored, {len(degraded)} degraded")

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "LongMemEval", "comparison": "graph (mix) vs flat (naive), one engine",
        "model": args.model, "judges": panel,
        "means": {m: (sum(tot[m])/len(tot[m]) if tot[m] else None) for m in MODES},
        "by_type": {t: {m: (sum(v)/len(v) if v else None) for m, v in d.items()}
                    for t, d in by_type.items()},
        "degraded": degraded, "reportable": not degraded,
        "instances": per_instance}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
