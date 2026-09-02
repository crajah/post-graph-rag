"""Replay a previous ECT-QA run's exact questions under the current engine.

run.py samples its question set, so two invocations do not share a set and
their headline accuracies are not comparable. This replays the questions,
spaces and gold answers recorded in a prior results file and scores them with
run.py's own logic, which makes the comparison paired: same graph, same
questions, same scoring, only the engine differs.
"""
import argparse
import asyncio
import collections
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from run import (ANSWER_INSTRUCTION, answer_similarity, numeric_f1,  # noqa: E402
                 period_f1)

from post_graph import RESERVED_SPACE_ALL                            # noqa: E402
from post_graph_rag import GraphRAG, QueryParam, RAGConfig           # noqa: E402
from post_graph_rag.llm import LLMService                            # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="prior results json to replay")
    ap.add_argument("--realm", default="ectqa_g36")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--top-k", type=int, default=48)
    ap.add_argument("--f1-threshold", type=float, default=0.60)
    ap.add_argument("--similarity-threshold", type=float, default=0.62)
    ap.add_argument("--query-concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.loads(pathlib.Path(args.baseline).read_text())
    old = base["results"]
    print(f"replaying {len(old)} questions from {pathlib.Path(args.baseline).name}"
          f" on realm {args.realm}\n", flush=True)

    rag = GraphRAG(RAGConfig(
        model=args.model, realm=args.realm, schema_per_realm=True,
        embedding_model=args.embedding_model, embedding_dim=1536,
        embed_relations=True, merge_strategy="rrf",
        pool_min_size=1, pool_max_size=4, max_retries=40))
    await rag.initialize()
    embed_llm = LLMService(RAGConfig(model=args.model,
                                     embedding_model=args.embedding_model,
                                     embedding_dim=1536))
    sem = asyncio.Semaphore(args.query_concurrency)
    results, degraded = [], []

    async def ask(rec):
        async with sem:
            t0 = time.time()
            try:
                out = await rag.query(
                    f"{ANSWER_INSTRUCTION}\n\nQuestion: {rec['question']}",
                    param=QueryParam(mode="mix", top_k=args.top_k,
                                     space=rec["space"]))
                answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()
            except Exception as e:
                degraded.append({"question": rec["question"][:80],
                                 "reason": f"{type(e).__name__}: {e}"})
                return
            gold = rec["gold"]
            if gold == "unanswerable":
                ok = ("unanswerable" in answer.lower()
                      or "not support" in answer.lower()
                      or "do not contain" in answer.lower())
                votes = {"rule": ok}
            else:
                f1 = numeric_f1(gold, answer)
                pf1 = period_f1(gold, answer) if f1 is None else None
                if f1 is not None:
                    ok = f1 >= args.f1_threshold; votes = {"numeric_f1": round(f1, 4)}
                elif pf1 is not None:
                    ok = pf1 >= args.f1_threshold; votes = {"period_f1": round(pf1, 4)}
                else:
                    sim = await answer_similarity(embed_llm, gold, answer)
                    ok = sim >= args.similarity_threshold; votes = {"cosine": round(sim, 4)}
            results.append({**{k: rec[k] for k in ("space", "companies", "type",
                                                   "question", "gold")},
                            "correct": bool(ok), "votes": votes,
                            "query_secs": round(time.time() - t0, 1),
                            "answer": answer[:800]})
            print(f"  {rec['type']:<18} {'CORRECT' if ok else 'incorrect':<9} "
                  f"[{len(results)}/{len(old)}]", flush=True)

    await asyncio.gather(*(ask(r) for r in old))
    await rag.close()

    prev = {r["question"]: r["correct"] for r in old}
    byt = collections.defaultdict(lambda: [0, 0, 0])
    for r in results:
        b = byt[r["type"]]
        b[0] += 1 if prev.get(r["question"]) else 0
        b[1] += 1 if r["correct"] else 0
        b[2] += 1
    print("\n" + "-" * 62)
    print(f"  {'type':<22}{'before':>8}{'after':>8}{'delta':>9}   n")
    to = tn = tt = 0
    for t in sorted(byt, key=lambda x: -byt[x][2]):
        o, n_, c = byt[t]; to += o; tn += n_; tt += c
        print(f"  {t:<22}{o/c:>8.2f}{n_/c:>8.2f}{(n_-o)/c:>+9.2f}   {c}")
    print(f"  {'OVERALL':<22}{to/tt:>8.2f}{tn/tt:>8.2f}{(tn-to)/tt:>+9.2f}   {tt}")
    up = sum(1 for r in results if r["correct"] and not prev.get(r["question"]))
    dn = sum(1 for r in results if not r["correct"] and prev.get(r["question"]))
    print(f"\n  flipped wrong->right: {up}   right->wrong: {dn}")
    print(f"  degraded: {len(degraded)}")

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "ECT-QA (paired replay)", "baseline": args.baseline,
        "realm": args.realm, "model": args.model, "n": len(results),
        "accuracy_before": to / tt if tt else None,
        "accuracy_after": tn / tt if tt else None,
        "by_type": {t: {"before": b[0]/b[2], "after": b[1]/b[2], "n": b[2]}
                    for t, b in byt.items()},
        "flipped_up": up, "flipped_down": dn,
        "degraded": degraded, "reportable": not degraded,
        "results": results}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
