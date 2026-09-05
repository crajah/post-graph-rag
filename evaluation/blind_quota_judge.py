"""Blind A/B comparison of relation_seed_quota 0.5 against 1.0.

The earlier measurement scored keyword overlap, which correlates with the
embedding similarity the relation channel ranks by — so the metric partly
rewarded the thing under test, and it could not separate 0.5 from 1.0. This
compares the answers instead.

Controls:

  blind        the judge sees two answers labelled A and B and the question,
               and is told nothing about how either was retrieved
  order        every pair is judged twice with the labels swapped; a win counts
               only if the judge picks the same answer both ways, so a judge
               with a positional preference produces no result rather than a
               wrong one. This is not a formality — on the two graphs measured
               it discarded 7 of 20 and 11 of 30 judgements.
  panel        several judges, none of them the model that wrote the answers,
               so a model cannot prefer its own prose

Answers are generated once per (question, quota) and reused across judges and
orders, so the only thing varying within a comparison is the retrieval setting.
"""
import argparse
import asyncio
import json
import pathlib
import re
import time

from post_graph_rag import GraphRAG, RAGConfig
from post_graph_rag.llm import LLMService
from post_graph_rag.models import QueryParam

# Three question shapes, because the two channels are expected to differ by
# shape rather than uniformly: traversal should be strongest where the question
# names a well-connected entity, and weakest where it names none.
QUESTIONS = [
    ("faa", "entity", "What role did the FAA play across these filings?"),
    ("737max", "entity", "What happened to the 737 MAX programme?"),
    ("bca revenue", "entity",
     "How did Boeing Commercial Airplanes' revenue change across these filings?"),
    ("dividend", "entity", "What did Boeing say about its dividend?"),
    ("deferred costs", "thematic",
     "How did deferred production costs change from a minor line item to a driver of cash burn?"),
    ("cash flow negative", "thematic", "What caused Boeing's cash flow to turn negative?"),
    ("risk factors", "thematic",
     "What risks did management flag as most likely to affect future results?"),
    ("supply chain", "thematic", "How did supply chain problems affect production?"),
    ("737 over time", "chain",
     "How did the relationship between the 737 programme and Boeing's cash flow change over time?"),
    ("regulation to money", "chain",
     "How did regulatory action translate into financial consequences?"),
]

JUDGE_PROMPT = """You are grading two answers to the same question about Boeing's
annual SEC filings. Both were written from relationships retrieved from a
knowledge graph built over those filings.

Judge only which answer is more useful to someone who asked the question:

- does it actually answer what was asked, rather than circling the topic
- is it specific — named programmes, line items, causes, periods
- is it supported, rather than hedging or padding
- length is not quality; a shorter answer that answers is better than a longer
  one that does not

---Question---
{question}

---Answer A---
{a}

---Answer B---
{b}

Reply with exactly one line:
VERDICT: A
VERDICT: B
VERDICT: TIE
then one short sentence of reason. Use TIE only when neither is meaningfully
better."""

VERDICT_RE = re.compile(r"VERDICT:\s*\**\s*(A|B|TIE)\b", re.IGNORECASE)


def parse_verdict(text):
    """Read the judge's choice, or None when the reply is unusable.

    Unusable replies are dropped rather than guessed at: a garbled judgement
    counted as a tie would quietly bias the result toward 'no difference'.
    """
    m = VERDICT_RE.search(text or "")
    return m.group(1).upper() if m else None


async def answer_for(realm, model, embedding_model, quota, questions, retries):
    """Answers for every question at one quota, through the shipped query path."""
    rag = GraphRAG(RAGConfig(
        model=model, realm=realm, schema_per_realm=True,
        embedding_model=embedding_model, embedding_dim=1536,
        embed_relations=True, relation_seed_quota=quota,
        max_retries=retries, retry_deadline_secs=900,
    ))
    await rag.initialize()
    out = {}
    try:
        for key, _shape, q in questions:
            res = await rag.query(q, param=QueryParam(mode="mix", top_k=6))
            ans = (res["answer"] if isinstance(res, dict) else str(res)).strip()
            out[key] = ans
            print(f"    [{quota}] {key:<20} {len(ans):>5} chars", flush=True)
    finally:
        await rag.close()
    return out


async def judge_pair(llm, question, first, second):
    """One judgement, in the A/B terms the judge saw.

    A judgement that never completes is recorded as unusable rather than
    aborting the run: the gateway in front of these models drops connections,
    and losing 40 completed judgements to the 41st is not a useful failure.
    """
    try:
        reply = await llm.chat_completion([{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, a=first, b=second)}])
    except Exception as e:
        return None, f"call failed: {type(e).__name__}"
    text = reply if isinstance(reply, str) else str(reply)
    return parse_verdict(text), text.strip()[:300]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realm", default="boeing_mm")
    ap.add_argument("--model", default="MiniMax-M2.7")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--judges", nargs="+", default=["gemma-4-31B-it", "DeepSeek-V3.2"])
    ap.add_argument("--quotas", nargs=2, type=float, default=[0.5, 1.0])
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--out", default="reports/blind_quota.json")
    args = ap.parse_args()

    lo, hi = args.quotas
    assert args.model not in args.judges, "a judge must not grade its own prose"

    started = time.time()
    print(f"realm={args.realm} model={args.model} judges={','.join(args.judges)}")

    # Answers are cached so a dropped connection during judging does not cost
    # another 20 generations. Keyed by realm and quota, so caches from different
    # graphs cannot be mistaken for each other.
    cache_path = pathlib.Path(f"{args.out}.answers")
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    answers = {}
    for quota in (lo, hi):
        ck = f"{args.realm}|{args.model}|{quota}"
        if ck in cached and set(cached[ck]) == {k for k, _s, _q in QUESTIONS}:
            print(f"  reusing cached answers at quota {quota}", flush=True)
            answers[quota] = cached[ck]
            continue
        print(f"  generating answers at quota {quota} ...", flush=True)
        answers[quota] = await answer_for(args.realm, args.model, args.embedding_model,
                                          quota, QUESTIONS, args.retries)
        cached[ck] = answers[quota]
        cache_path.write_text(json.dumps(cached, indent=2))

    print("\n  judging (each pair twice, labels swapped)\n", flush=True)
    print(f"  {'question':<20} {'shape':<9} {'judge':<16} {'fwd':>4} {'rev':>4}  result")
    print("  " + "-" * 72)

    judge_llms = {j: LLMService(RAGConfig(model=j, max_retries=args.retries,
                                          retry_deadline_secs=900))
                  for j in args.judges}

    records, tally = [], {}
    for key, shape, q in QUESTIONS:
        a_lo, a_hi = answers[lo][key], answers[hi][key]
        for judge in args.judges:
            # forward: A is the low quota. reverse: A is the high quota.
            fwd, fwd_txt = await judge_pair(judge_llms[judge], q, a_lo, a_hi)
            rev, rev_txt = await judge_pair(judge_llms[judge], q, a_hi, a_lo)

            # Translate both into which quota was preferred.
            pick_fwd = {"A": lo, "B": hi, "TIE": "tie"}.get(fwd or "")
            pick_rev = {"A": hi, "B": lo, "TIE": "tie"}.get(rev or "")
            if pick_fwd is None or pick_rev is None:
                result = "unusable"
            elif pick_fwd == pick_rev and pick_fwd != "tie":
                result = f"quota {pick_fwd}"
            elif pick_fwd == "tie" and pick_rev == "tie":
                result = "tie"
            else:
                result = "inconsistent"  # judge followed position, not content
            tally[result] = tally.get(result, 0) + 1
            print(f"  {key:<20} {shape:<9} {judge:<16} {str(fwd):>4} {str(rev):>4}  {result}",
                  flush=True)
            records.append({"question": key, "shape": shape, "judge": judge,
                            "forward": fwd, "reverse": rev, "result": result,
                            "forward_reason": fwd_txt, "reverse_reason": rev_txt})

    print("\n  tally:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print(f"  {time.time() - started:.0f}s")
    pathlib.Path(args.out).write_text(json.dumps(
        {"realm": args.realm, "model": args.model, "judges": args.judges,
         "quotas": [lo, hi], "answers": {str(k): v for k, v in answers.items()},
         "judgements": records, "tally": tally}, indent=2))
    print(f"  wrote {args.out}")


asyncio.run(main())
