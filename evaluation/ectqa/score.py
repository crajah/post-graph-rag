"""Score an ECT-QA results file by numeric F1, with cosine as the fallback.

Rescoring an existing run costs nothing but embedding calls for the narrative
minority: run.py already stores the gold answer and the system's answer for
every question, so a metric change does not require re-indexing.

Why not the judge panel this replaces: on "gross margin in each quarter of 2022"
the system answered 33.9%, 33.6%, 31.7% and 32% against a gold of 33.9%, 33.6%,
31.7%, 32.3% — three exact, one within 0.3 points — and all three judges marked
the whole answer wrong. Measured across a full run, correct answers and refusals
were 0.643 and 0.509 apart under whole-text cosine, against 0.770 and 0.144
under figure matching.

Usage:
    python evaluation/ectqa/score.py results_g36.json [--f1-threshold 0.6]
"""
import argparse
import asyncio
import collections
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

_spec = importlib.util.spec_from_file_location("_ectqa_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_run)
except SystemExit:
    pass
numeric_f1 = _run.numeric_f1
period_f1 = _run.period_f1

from post_graph_rag import RAGConfig                      # noqa: E402
from post_graph_rag.llm import LLMService                 # noqa: E402


def is_refusal(answer: str) -> bool:
    return (answer or "").strip().lower()[:60].startswith("unanswerable")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if not na or not nb else dot / (na * nb)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--f1-threshold", type=float, default=0.60)
    ap.add_argument("--similarity-threshold", type=float, default=0.62)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_absolute():
        path = HERE / path
    data = json.loads(path.read_text())
    rows = data["results"]
    gold_rows = [r for r in rows if r.get("gold") != "unanswerable"]
    unans_rows = [r for r in rows if r.get("gold") == "unanswerable"]

    llm = LLMService(RAGConfig(embedding_model=args.embedding_model))
    scored, by_type = [], collections.defaultdict(list)
    for r in gold_rows:
        gold, answer = str(r["gold"]), str(r.get("answer") or "")
        f1 = numeric_f1(gold, answer)
        pf1 = period_f1(gold, answer) if f1 is None else None
        if f1 is not None:
            ok, metric, value = f1 >= args.f1_threshold, "numeric_f1", f1
        elif pf1 is not None:
            ok, metric, value = pf1 >= args.f1_threshold, "period_f1", pf1
        else:
            sim = cosine(await llm.get_embedding(gold), await llm.get_embedding(answer)) \
                if answer.strip() else 0.0
            ok, metric, value = sim >= args.similarity_threshold, "cosine", sim
        scored.append({**r, "metric": metric, "score": round(value, 4),
                       "correct": bool(ok), "refusal": is_refusal(answer)})
        by_type[r["type"]].append(bool(ok))

    # Declining correctly on an unanswerable question is still a correct outcome.
    for r in unans_rows:
        ok = is_refusal(str(r.get("answer") or ""))
        scored.append({**r, "metric": "refusal_rule", "score": 1.0 if ok else 0.0,
                       "correct": ok, "refusal": ok})
        by_type["unanswerable"].append(ok)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    total = mean([s["correct"] for s in scored])
    fig = [s for s in scored if s["metric"] == "numeric_f1"]
    refused = [s for s in gold_rows if is_refusal(str(s.get("answer") or ""))]

    print(f"\n{path.name}   model={data.get('model')}   n={len(scored)}")
    print(f"  accuracy (numeric F1 primary)  {total:.3f}")
    print(f"  mean numeric F1 over {len(fig):>3} figure questions   {mean([s['score'] for s in fig]):.3f}")
    print(f"  refused outright               {len(refused)}/{len(gold_rows)} "
          f"({len(refused)/max(1,len(gold_rows)):.0%})")
    print("  by type:")
    for t, v in sorted(by_type.items()):
        print(f"    {t:22s} {mean(v):.3f}  (n={len(v)})")

    out = path.with_name(path.stem + "_scored.json")
    out.write_text(json.dumps({
        "benchmark": "ECT-QA", "metric": "numeric F1 with tolerance; cosine fallback",
        "source": path.name, "model": data.get("model"),
        "f1_threshold": args.f1_threshold, "n": len(scored),
        "accuracy": round(total, 4),
        "mean_numeric_f1": round(mean([s["score"] for s in fig]), 4),
        "refused": len(refused),
        "by_type": {t: round(mean(v), 4) for t, v in by_type.items()},
        "results": scored,
    }, indent=1))
    print(f"  wrote {out.name}")


asyncio.run(main())
