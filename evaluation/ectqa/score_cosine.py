"""Score ECT-QA by cosine similarity between the gold answer and the system's.

ECT-QA answers are lists of figures — "$107.3 million, $125.3 million, $96.4
million" — and a judge panel asked whether that is "correct" ends up arguing
about rounding, ordering and units rather than measuring retrieval. Embedding
similarity against the gold string is the benchmark's own metric and does not
have opinions.

Two things follow from the metric that are worth stating rather than hiding:

  * A refusal scores low but not zero, because "the information is not available"
    still shares vocabulary with a financial answer. Refusals are therefore
    reported separately — a mean similarity that quietly averages in refusals
    describes the embedding space, not the system.
  * A threshold turns similarity into accuracy and is a choice, not a fact.
    Both are reported: the mean similarity, which needs no threshold, and the
    accuracy at whatever threshold you pass.

Usage:
    python evaluation/ectqa/score_cosine.py results_gemma4.json [--threshold 0.8]
"""
import argparse
import asyncio
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from post_graph_rag import RAGConfig                      # noqa: E402
from post_graph_rag.llm import LLMService                 # noqa: E402

def is_refusal(answer: str) -> bool:
    """Only a leading refusal counts.

    Matching those phrases anywhere was wrong: a good answer that ends with an
    honest caveat — "the results for other quarters are missing" — was counted
    as a refusal, which put 50 of 60 in that bucket while several of them stated
    the gold figures correctly. The harness itself judges refusal by what the
    answer opens with, and so does this.
    """
    head = (answer or "").strip().lower()[:60]
    return head.startswith("unanswerable")


FIGURE = re.compile(r"-?\d+(?:\.\d+)?%?")


def figure_recall(gold: str, answer: str):
    """Share of the gold figures that appear in the answer.

    Cosine on this benchmark is depressed by verbosity: gold is a bare list
    ("33.9%, 33.6%, 31.7%, and 32.3%") while the system replies in prose. Two
    answers containing exactly the same figures score differently by length
    alone. Figure recall is unaffected by wording and is reported alongside.
    """
    g = [x for x in FIGURE.findall(gold or "") if any(c.isdigit() for c in x)]
    if not g:
        return None
    a = set(FIGURE.findall(answer or ""))
    hit = sum(1 for x in g if x in a or x.rstrip('%') in {y.rstrip('%') for y in a})
    return hit / len(g)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if not na or not nb else dot / (na * nb)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a results JSON written by run.py")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="similarity at or above which an answer counts as correct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_absolute():
        path = HERE / path
    data = json.loads(path.read_text())
    rows = [r for r in data["results"] if r.get("gold") != "unanswerable"]
    print(f"{path.name}: {len(data['results'])} results, "
          f"{len(rows)} with a gold answer to compare against")

    llm = LLMService(RAGConfig(embedding_model=args.embedding_model))
    scored = []
    for r in rows:
        gold, answer = str(r["gold"]), str(r.get("answer") or "")
        if not answer.strip():
            sim = 0.0
        else:
            ge, ae = await llm.get_embedding(gold), await llm.get_embedding(answer)
            sim = cosine(ge, ae)
        scored.append({**r, "similarity": round(sim, 4), "refusal": is_refusal(answer),
                       "figure_recall": figure_recall(gold, answer)})

    answered = [s for s in scored if not s["refusal"]]
    refused = [s for s in scored if s["refusal"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n  refused        {len(refused):>3}/{len(scored)}  "
          f"({len(refused)/len(scored):.0%})   mean similarity {mean([s['similarity'] for s in refused]):.3f}")
    print(f"  answered       {len(answered):>3}/{len(scored)}  "
          f"({len(answered)/len(scored):.0%})   mean similarity {mean([s['similarity'] for s in answered]):.3f}")
    print(f"  all            {len(scored):>3}          mean similarity "
          f"{mean([s['similarity'] for s in scored]):.3f}")
    fr = [s["figure_recall"] for s in scored if s["figure_recall"] is not None]
    exact = [x for x in fr if x == 1.0]
    print(f"\n  figure recall  mean {mean(fr):.3f} over {len(fr)} gold answers containing figures")
    print(f"                 all figures present in {len(exact)}/{len(fr)} = {len(exact)/len(fr):.1%}")
    hits = [s for s in scored if s["similarity"] >= args.threshold]
    print(f"\n  accuracy @ {args.threshold:.2f}: {len(hits)}/{len(scored)} = "
          f"{len(hits)/len(scored):.1%}   (threshold is a choice, not a measurement)")

    by_type = {}
    for s in scored:
        by_type.setdefault(s["type"], []).append(s["similarity"])
    print("\n  mean similarity by type:")
    for t, sims in sorted(by_type.items()):
        print(f"    {t:22s} {mean(sims):.3f}  (n={len(sims)})")

    out = pathlib.Path(args.out) if args.out else path.with_name(path.stem + "_cosine.json")
    out.write_text(json.dumps({
        "benchmark": "ECT-QA", "metric": "cosine similarity to gold answer",
        "embedding_model": args.embedding_model, "threshold": args.threshold,
        "source": path.name, "n": len(scored),
        "refused": len(refused), "answered": len(answered),
        "mean_similarity": round(mean([s["similarity"] for s in scored]), 4),
        "mean_similarity_answered": round(mean([s["similarity"] for s in answered]), 4),
        "accuracy_at_threshold": round(len(hits) / len(scored), 4) if scored else 0.0,
        "figure_recall_mean": round(mean(fr), 4) if fr else None,
        "figure_recall_complete": round(len(exact) / len(fr), 4) if fr else None,
        "by_type": {t: round(mean(v), 4) for t, v in by_type.items()},
        "results": scored,
    }, indent=1))
    print(f"\n  wrote {out.name}")


asyncio.run(main())
