"""ECT-QA scored the way the benchmark's authors score it.

TG-RAG (arXiv:2510.13590), which introduced ECT-QA, does not use figure
matching as its headline. It has an LLM judge perform a "fine-grained,
element-wise comparison between the model's prediction and the ground-truth
answer", reporting three rates that sum to one per query:

    Correct    claims accurately supported and matching the temporal scope
    Refusal    the model explicitly acknowledges it cannot answer
    Incorrect  wrong, unsupported or hallucinated content

Two differences from this repository's strict scorer matter, and they run in
opposite directions from the usual worry about LLM judges:

  * It is element-wise, not per-question. Three of four quarters right scores
    0.75, where strict thresholding scores zero.
  * Refusal is its own outcome. A model that declines for want of evidence is
    not counted as wrong, which is the correct treatment for a corpus where
    some questions genuinely cannot be answered from the transcripts.

This scores STORED answers from a previous run rather than generating new
ones, so the native and strict protocols describe exactly the same
generations and can be reported side by side.

Deviations from the paper, stated so nobody has to guess:
  * Their judge is GPT-4o-mini. It is not on this router; gpt-4o is, and is
    used here -- same family, stronger grader.
  * Their verbatim prompt is in an appendix that is truncated in the public
    HTML. The rubric below follows their description of the three categories
    but is not their text, so absolute comparability is approximate.
"""
import argparse
import asyncio
import collections
import json
import pathlib
import sys
from typing import List

from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from post_graph_rag import RAGConfig                     # noqa: E402
from post_graph_rag.llm import LLMService                # noqa: E402


class ElementVerdict(BaseModel):
    element: str = Field(description="One atomic fact from the ground truth")
    verdict: str = Field(description="correct | refusal | incorrect")


class Judgement(BaseModel):
    elements: List[ElementVerdict]


RUBRIC = """You are grading an answer about earnings call transcripts.

Decompose the GROUND TRUTH into its atomic factual elements -- each figure,
each named quarter, each named company. For every element, judge how the
PREDICTION treats it:

  correct    the prediction states this element accurately, with the temporal
             scope the question asks for. Figures within about 1% agree.
             Different wording or ordering is fine. A prediction that states
             the element correctly is correct even if it also explains its
             reasoning or names other periods for comparison.
  refusal    the prediction explicitly says it cannot answer this element, or
             that the evidence does not contain it. Declining is a refusal,
             never an incorrect answer.
  incorrect  the prediction states a wrong value for this element, or asserts
             something the ground truth contradicts, or fabricates it.

Return one verdict per ground-truth element and nothing else.

QUESTION:
{question}

GROUND TRUTH:
{gold}

PREDICTION:
{prediction}"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="a prior results file whose stored answers are re-scored")
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.loads(pathlib.Path(args.results).read_text())
    rows = src["results"]
    judge = LLMService(RAGConfig(model=args.judge, max_retries=25,
                                 retry_deadline_secs=900))
    print(f"{len(rows)} stored answers | judge {args.judge} | "
          f"element-wise Correct/Refusal/Incorrect\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    scored, failed = [], []

    async def one(r):
        async with sem:
            gold, ans = r["gold"], r.get("answer") or ""
            if gold == "unanswerable":
                # The dataset's own unanswerable questions: declining IS the
                # correct answer, so a refusal scores correct rather than
                # landing in its own bucket and deflating every rate.
                low = ans.lower()
                declined = ("unanswerable" in low or "not support" in low
                            or "do not contain" in low or "no context" in low)
                scored.append({**{k: r[k] for k in ("type", "question")},
                               "correct": 1.0 if declined else 0.0,
                               "refusal": 0.0,
                               "incorrect": 0.0 if declined else 1.0,
                               "elements": 1})
                return
            try:
                out = await judge.chat_completion(
                    [{"role": "user", "content": RUBRIC.format(
                        question=r["question"], gold=gold, prediction=ans)}],
                    response_format=Judgement)
            except Exception as e:
                failed.append({"question": r["question"][:80],
                               "reason": f"{type(e).__name__}: {e}"})
                return
            els = out.elements if hasattr(out, "elements") else []
            if not els:
                failed.append({"question": r["question"][:80],
                               "reason": "judge returned no elements"})
                return
            c = collections.Counter(e.verdict.strip().lower() for e in els)
            n = len(els)
            scored.append({**{k: r[k] for k in ("type", "question")},
                           "correct": c["correct"] / n,
                           "refusal": c["refusal"] / n,
                           "incorrect": c["incorrect"] / n,
                           "elements": n})
            print(f"  {r['type']:<20} correct={c['correct']}/{n} "
                  f"refusal={c['refusal']} incorrect={c['incorrect']}", flush=True)

    await asyncio.gather(*(one(r) for r in rows))

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in scored:
        for k in ("correct", "refusal", "incorrect"):
            by[s["type"]][k].append(s[k])
    print("\n" + "=" * 62)
    print(f"  {'type':<22}{'Correct':>10}{'Refusal':>10}{'Incorrect':>11}{'n':>5}")
    for t in sorted(by):
        d = by[t]
        print(f"  {t:<22}{sum(d['correct'])/len(d['correct']):>10.3f}"
              f"{sum(d['refusal'])/len(d['refusal']):>10.3f}"
              f"{sum(d['incorrect'])/len(d['incorrect']):>11.3f}"
              f"{len(d['correct']):>5}")
    tot = {k: [s[k] for s in scored] for k in ("correct", "refusal", "incorrect")}
    print(f"  {'OVERALL':<22}{sum(tot['correct'])/len(tot['correct']):>10.3f}"
          f"{sum(tot['refusal'])/len(tot['refusal']):>10.3f}"
          f"{sum(tot['incorrect'])/len(tot['incorrect']):>11.3f}"
          f"{len(scored):>5}")
    print(f"\n  judge failures: {len(failed)}")
    print("  reference: TG-RAG 0.599 correct, GraphRAG 0.405, LightRAG 0.406 "
          "(their judge, their corpus slice)")

    pathlib.Path(args.out).write_text(json.dumps({
        "benchmark": "ECT-QA (native element-wise protocol)",
        "source_results": args.results, "judge": args.judge,
        "judge_in_paper": "gpt-4o-mini (unavailable on this router)",
        "protocol": "element-wise Correct/Refusal/Incorrect, rates sum to 1 per query",
        "rubric_is_verbatim_from_paper": False,
        "means": {k: sum(v) / len(v) for k, v in tot.items()},
        "by_type": {t: {k: sum(v) / len(v) for k, v in d.items()}
                    for t, d in by.items()},
        "failed": failed, "reportable": not failed,
        "results": scored}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
