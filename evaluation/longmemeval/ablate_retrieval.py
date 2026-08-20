"""Paired ablation of the retrieval-side features, on one graph per instance.

`run.py --mmr` and friends re-index for every arm, which makes the comparison
unpaired: each arm gets its own graph, built by a nondeterministic model, and
then answers with a nondeterministic model. Run that way, the baseline moved
five points between two runs of identical code — the same size as the effects
being measured. At twenty instances one flipped answer is five points, so a
difference that size says nothing.

This indexes each instance once and queries the same graph with each variant in
turn. Indexing variance cancels, because every arm sees byte-identical data.
What remains is answering nondeterminism, which is why each variant is asked
`--repeats` times and the vote is taken across them.

Contradiction detection cannot be measured this way — it changes what is
written, not what is read — so it stays in run.py.
"""
import argparse
import asyncio
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

from post_graph_rag import GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402

# Each arm is a name and the config overrides that define it. Applied to a
# single live GraphRAG between queries, which works because retrieval reads
# these flags at query time rather than at construction.
VARIANTS = {
    "baseline":     {"mmr_enabled": False, "node_distance_rerank": False},
    "mmr":          {"mmr_enabled": True,  "node_distance_rerank": False},
    "nodedistance": {"mmr_enabled": False, "node_distance_rerank": True},
    "both":         {"mmr_enabled": True,  "node_distance_rerank": True},
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "oracle.json"))
    ap.add_argument("--model", default="google/gemma-4-26b-a4b-it-maas")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--judges", nargs="*",
                    default=["MiniMax-M2.7", "gpt-oss-120b", "DeepSeek-V3.2"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--types", nargs="*",
                    default=["temporal-reasoning", "knowledge-update"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--chunk-concurrency", type=int, default=8)
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    # Answering is nondeterministic; one sample per arm cannot separate a real
    # effect from a resample of the same distribution.
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default=str(HERE / "ablation_paired.json"))
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text())
    data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900))
                  for n in args.judges}

    print(f"{len(data)} instances | one graph each | model={args.model}")
    print(f"variants: {', '.join(args.variants)} | repeats: {args.repeats}")
    print(f"judges: {', '.join(args.judges)}\n")

    # correct[variant] = list of per-instance scores in [0, 1] (mean over repeats)
    scores: dict = defaultdict(list)
    per_instance: list = []
    degraded: list = []
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()

    async def one(i, inst):
        async with sem:
            rag = GraphRAG(RAGConfig(
                model=args.model, realm=f"pair_{args.seed}_{i}", schema_per_realm=True,
                embedding_model=args.embedding_model, embedding_dim=1536,
                embed_relations=True, merge_strategy="rrf",
                max_concurrent_chunks=args.chunk_concurrency,
                mmr_lambda=args.mmr_lambda,
                extraction_prompt=CONVERSATIONAL_PROMPT,
                max_retries=40, retry_deadline_secs=1800))
            await rag.initialize()
            try:
                try:
                    await index_instance(rag, inst)
                except DegradedRun as e:
                    degraded.append({"question_id": inst["question_id"], "reason": str(e)})
                    print(f"{inst['question_id']:<16} SKIPPED ({str(e)[:44]})", flush=True)
                    return

                asked = (f"Today is {parse_date(inst.get('question_date', ''))}. "
                         f"{inst['question']}") if inst.get("question_date") else inst["question"]

                row = {"question_id": inst["question_id"], "type": inst["question_type"],
                       "gold": inst["answer"], "variants": {}}
                for name in args.variants:
                    for key, value in VARIANTS[name].items():
                        setattr(rag.config, key, value)
                    marks, answers = [], []
                    for _ in range(args.repeats):
                        out = await rag.query(asked, param=QueryParam(mode="mix", top_k=8))
                        answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()
                        ok, _votes = await judge_panel(judge_llms, inst["question"],
                                                       inst["answer"], answer)
                        marks.append(bool(ok))
                        answers.append(answer[:400])
                    score = sum(marks) / len(marks)
                    scores[name].append(score)
                    row["variants"][name] = {"score": score, "marks": marks,
                                            "answers": answers}
                per_instance.append(row)
                summary = "  ".join(f"{n}={row['variants'][n]['score']:.2f}"
                                    for n in args.variants)
                print(f"{inst['question_id']:<16} {summary}", flush=True)
            except Exception as e:
                degraded.append({"question_id": inst["question_id"],
                                 "reason": f"{type(e).__name__}: {e}"})
                print(f"{inst['question_id']:<16} FAILED {type(e).__name__}: "
                      f"{str(e)[:50]}", flush=True)
            finally:
                await rag.close()

    await asyncio.gather(*(one(i, inst) for i, inst in enumerate(data)))

    print("\n" + "-" * 62)
    base = scores.get("baseline", [])
    for name in args.variants:
        vals = scores[name]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        line = f"  {name:<14} {mean*100:5.1f}%  n={len(vals)}"
        if name != "baseline" and base:
            # Paired: same instance, same graph, so the per-instance difference
            # is the quantity of interest, not the difference of the means.
            deltas = [v - b for v, b in zip(vals, base)]
            better = sum(1 for d in deltas if d > 0)
            worse = sum(1 for d in deltas if d < 0)
            line += (f"   delta={sum(deltas)/len(deltas)*100:+5.1f} pts"
                     f"  better={better} worse={worse} same={len(deltas)-better-worse}")
        print(line)
    if degraded:
        print(f"\n  !! {len(degraded)} of {len(data)} instances degraded — not reportable")
    print(f"  {time.time() - started:.0f}s total")

    pathlib.Path(args.out).write_text(json.dumps(
        {"model": args.model, "judges": args.judges, "repeats": args.repeats,
         "variants": {n: VARIANTS[n] for n in args.variants},
         "mmr_lambda": args.mmr_lambda,
         "means": {n: (sum(v) / len(v) if v else None) for n, v in scores.items()},
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded, "instances": per_instance}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
