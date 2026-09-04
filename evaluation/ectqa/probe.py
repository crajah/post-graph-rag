"""ECT-QA arms probe: separate what the metric costs us from what the engine does.

Replays one prior run's exact questions under several arms and scores every
arm two ways -- over the whole answer (the established metric) and over a
final "ANSWER:" line when the arm asks for one. Reporting both is the point:
a gain that appears only under the new metric is a measurement artefact, and
a gain that survives both is real.

Arms
  control     current instruction, current scoring. Reproduces the baseline.
  answerline  same retrieval; the model must end with a bare figures line.
              Tests the measurement hypothesis: 13% of questions found every
              gold figure yet scored wrong because surrounding prose carries
              numbers, and precision counts them against the answer.
  decompose   auto_decompose on. A multi-time question ("each quarter of
              2022") is four retrievals wearing one sentence; one embedding
              of the whole question resembles no single quarter's facts.
  scatter     cross-company questions only: ask each company's space
              separately, then merge. One blended search over the union lets
              the best-matching company consume the budget, which is the
              shape of a deficit that has survived every other explanation.
"""
import argparse
import asyncio
import collections
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from run import (ANSWER_INSTRUCTION, judge_panel, numeric_f1,  # noqa: E402
                 period_f1)

from post_graph import RESERVED_SPACE_ALL                            # noqa: E402
from post_graph_rag import GraphRAG, QueryParam, RAGConfig           # noqa: E402
from post_graph_rag.llm import LLMService                            # noqa: E402

ANSWER_LINE = (
    "\n\nFinally, after your explanation, end your reply with a single line:\n"
    "ANSWER: <only the figures asked for, comma-separated, in the order asked; "
    "no words, no periods, no citations>\n"
    "If a period's figure is missing, write NA in its place. This line is read "
    "by a scorer, so it must contain the asked values and nothing else."
)

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.I | re.S)


def answer_line(text: str) -> str:
    """The final ANSWER: line, or the whole text when the model omitted it."""
    hits = _ANSWER_RE.findall(text or "")
    return hits[-1].strip().splitlines()[0] if hits else (text or "")


def score(gold: str, ans: str, f1_t: float):
    """run.py's tiering, minus the cosine fallback (handled by the caller)."""
    if gold == "unanswerable":
        low = ans.lower()
        return bool("unanswerable" in low or "not support" in low
                    or "do not contain" in low), {"rule": True}
    f1 = numeric_f1(gold, ans)
    if f1 is not None:
        return f1 >= f1_t, {"numeric_f1": round(f1, 4)}
    pf1 = period_f1(gold, ans)
    if pf1 is not None:
        return pf1 >= f1_t, {"period_f1": round(pf1, 4)}
    return None, {}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--realm", default="ectqa_g36")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--arms", nargs="*",
                    default=["control", "answerline", "decompose", "scatter"])
    ap.add_argument("--top-k", type=int, default=48)
    ap.add_argument("--f1-threshold", type=float, default=0.60)
    ap.add_argument("--judges", nargs="*", default=["gemini-3.7-flash"])
    ap.add_argument("--query-concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old = json.loads(pathlib.Path(args.baseline).read_text())["results"]
    print(f"{len(old)} questions | arms: {', '.join(args.arms)} | realm {args.realm}\n",
          flush=True)

    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900))
                  for n in args.judges}
    rows = collections.defaultdict(dict)     # question -> arm -> record
    sem = asyncio.Semaphore(args.query_concurrency)

    def make_rag(decompose: bool):
        return GraphRAG(RAGConfig(
            model=args.model, realm=args.realm, schema_per_realm=True,
            embedding_model=args.embedding_model, embedding_dim=1536,
            embed_relations=True, merge_strategy="rrf", auto_decompose=decompose,
            pool_min_size=1, pool_max_size=4, max_retries=40))

    for arm in args.arms:
        rag = make_rag(decompose=(arm == "decompose"))
        await rag.initialize()
        instruction = ANSWER_INSTRUCTION + (ANSWER_LINE if arm != "control" else "")
        started = time.time()

        async def ask(rec):
            async with sem:
                q = rec["question"]
                try:
                    if arm == "scatter" and rec["space"] == RESERVED_SPACE_ALL \
                            and rec["gold"] != "unanswerable" and rec["companies"] != "-":
                        firms = [c for c in rec["companies"].split("+") if c]
                        parts = await asyncio.gather(*[
                            rag.query(f"{instruction}\n\nQuestion: {q}",
                                      param=QueryParam(mode="mix", top_k=args.top_k,
                                                       space=f))
                            for f in firms], return_exceptions=True)
                        chunks = []
                        for f, p in zip(firms, parts):
                            if isinstance(p, Exception):
                                continue
                            chunks.append(f"[{f}] " +
                                          (p["answer"] if isinstance(p, dict) else str(p)))
                        if not chunks:
                            return
                        merged = await rag.llm.chat_completion([{
                            "role": "user",
                            "content": (f"{instruction}\n\nQuestion: {q}\n\n"
                                        "Per-company findings follow. Combine them "
                                        "into one answer covering every company "
                                        "asked about.\n\n" + "\n\n".join(chunks))}])
                        answer = (merged if isinstance(merged, str)
                                  else str(merged)).strip()
                    else:
                        out = await rag.query(
                            f"{instruction}\n\nQuestion: {q}",
                            param=QueryParam(mode="mix", top_k=args.top_k,
                                             space=rec["space"]))
                        answer = (out["answer"] if isinstance(out, dict)
                                  else str(out)).strip()
                except Exception as e:
                    print(f"  {arm}: FAILED {type(e).__name__}: {str(e)[:50]}", flush=True)
                    return
                full_ok, full_v = score(rec["gold"], answer, args.f1_threshold)
                line = answer_line(answer)
                line_ok, line_v = score(rec["gold"], line, args.f1_threshold)
                if full_ok is None:              # narrative gold: judge reads it
                    full_ok, full_v = await judge_panel(
                        judge_llms, q, rec["gold"], answer)
                    line_ok, line_v = full_ok, full_v
                rows[q][arm] = {"type": rec["type"], "gold": rec["gold"],
                                "full_correct": bool(full_ok),
                                "line_correct": bool(line_ok),
                                "full_votes": full_v, "line_votes": line_v,
                                "answer": answer[:600]}

        await asyncio.gather(*(ask(r) for r in old))
        await rag.close()
        n = sum(1 for q in rows if arm in rows[q])
        fc = sum(1 for q in rows if arm in rows[q] and rows[q][arm]["full_correct"])
        lc = sum(1 for q in rows if arm in rows[q] and rows[q][arm]["line_correct"])
        print(f"{arm:<12} n={n:<4} full={fc/n:.3f}  answer-line={lc/n:.3f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    # ---- report ----
    prev = {r["question"]: r["correct"] for r in old}
    types = sorted({r["type"] for r in old})
    print("\n" + "=" * 74)
    for metric in ("full_correct", "line_correct"):
        print(f"\n  scored over: {'whole answer' if metric=='full_correct' else 'ANSWER: line'}")
        print(f"  {'type':<22}{'before':>8}" + "".join(f"{a:>12}" for a in args.arms))
        for t in types:
            qs = [q for q in rows if any(rows[q].get(a, {}).get("type") == t
                                         for a in args.arms)]
            if not qs:
                continue
            b = sum(1 for q in qs if prev.get(q)) / len(qs)
            line = f"  {t:<22}{b:>8.2f}"
            for a in args.arms:
                v = [rows[q][a][metric] for q in qs if a in rows[q]]
                line += f"{(sum(v)/len(v) if v else float('nan')):>12.2f}"
            print(line)
        allq = list(rows)
        b = sum(1 for q in allq if prev.get(q)) / len(allq)
        line = f"  {'OVERALL':<22}{b:>8.2f}"
        for a in args.arms:
            v = [rows[q][a][metric] for q in allq if a in rows[q]]
            line += f"{(sum(v)/len(v) if v else float('nan')):>12.2f}"
        print(line)

    pathlib.Path(args.out).write_text(json.dumps(
        {"benchmark": "ECT-QA arms probe", "baseline": args.baseline,
         "arms": args.arms, "realm": args.realm, "model": args.model,
         "rows": rows}, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
