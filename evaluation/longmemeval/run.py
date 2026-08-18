"""LongMemEval against post-graph-rag.

The benchmark is conversational — timestamped chat sessions, then a question
whose answer lives somewhere in them. post-graph-rag has only ever been
evaluated on document corpora, so the mapping is a design decision rather than
a formality, and a bad one would measure the harness instead of the engine.

Mapping used here:

  session -> document      One document per session, its turns rendered as
                           "user:"/"assistant:" lines. A session is the unit
                           the dataset timestamps, so it is the unit that can
                           carry a date.
  session date -> metadata The date reaches DocumentMetadata, so extraction can
                           attach validity and supersession can order two
                           contradicting sessions by when they happened rather
                           than by the order they were indexed.
  question -> query        Asked through query(), the shipped path, so the
                           number describes the library rather than a
                           reconstruction of it.

Every instance gets its own realm, because instances are independent haystacks
and one shared graph would let session facts from instance A answer instance B.
That is slower and it is the only honest arrangement.

Scoring is an LLM judge given the question, the gold answer and the produced
answer — the benchmark's own method. The judge model is held distinct from the
answering model so a model cannot mark its own work.
"""
import argparse
import asyncio
import json
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

JUDGE_PROMPT = """You are grading an answer against the reference answer for a
question about a user's chat history.

Mark CORRECT if the answer contains the same essential information as the
reference, even if worded differently or with extra detail. Mark INCORRECT if it
contradicts the reference, omits the essential fact, or declines to answer when
the reference gives one.

Question: {question}
Reference answer: {gold}
Answer to grade: {answer}

Reply with exactly one word: CORRECT or INCORRECT."""


def render_session(turns, date: str = "") -> str:
    """A session as text, dated in the body.

    The date is written into the text rather than left in metadata. Every
    temporal-reasoning question in this benchmark is of the form "how many weeks
    between X and Y" or "which happened first", so the session date is not
    context for the answer — it *is* the answer. Metadata never reaches the
    prompt, and a first run scored 0/3 with the model replying, correctly, that
    the retrieved context contained no dates.
    """
    header = f"[Conversation on {date}]\n" if date else ""
    return header + "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def parse_date(raw: str) -> str:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10'. Returns '' if unrecognised."""
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", raw or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


class DegradedRun(RuntimeError):
    """Extraction failed for part of an instance, so its graph is incomplete.

    Raised rather than tolerated. A partial graph answers "I cannot determine"
    for a question whose evidence simply was not indexed, which is
    indistinguishable from a genuine reasoning failure — and an accuracy figure
    that silently mixes the two measures the LLM provider's billing state as
    much as the engine. An earlier run scored 0/3 while the router was
    returning 402s, and nothing in the output said so.
    """


async def index_instance(rag, instance) -> int:
    """One document per session, in chronological order.

    Order matters: supersession resolves a contradiction by document order, so
    indexing a later session first would invert which fact wins.

    Raises DegradedRun if any session fails to index or yields nothing.
    """
    sessions = list(zip(instance["haystack_dates"],
                        instance["haystack_session_ids"],
                        instance["haystack_sessions"]))
    sessions.sort(key=lambda s: parse_date(s[0]) or s[0])
    for raw_date, session_id, turns in sessions:
        text = render_session(turns, parse_date(raw_date))
        if not text.strip():
            continue
        try:
            written = await rag.index_document(text, metadata=DocumentMetadata(
                document=session_id, source=f"session://{session_id}",
                category="chat_session", extra={"session_date": parse_date(raw_date)}))
        except Exception as e:
            raise DegradedRun(
                f"session {session_id} failed to index: {type(e).__name__}: {e}") from e
        if not written:
            raise DegradedRun(
                f"session {session_id} produced no graph structure; the evidence "
                f"for this question is not in the graph, so any answer would be "
                f"scored against an index that was never built")
    return len(sessions)


async def judge(llm, question, gold, answer) -> bool:
    reply = await llm.chat_completion([{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, gold=gold, answer=answer)}])
    text = (reply if isinstance(reply, str) else str(reply)).strip().upper()
    # Only an explicit CORRECT counts. An unparseable reply is not a pass.
    return text.startswith("CORRECT") or "CORRECT" in text[:40] and "INCORRECT" not in text[:40]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "oracle.json"))
    ap.add_argument("--model", default="google/gemma-4-26b-a4b-it-maas")
    ap.add_argument("--judge-model", default="MiniMax-M2.7")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--types", nargs="*", default=None,
                    help="question_type filter, e.g. temporal-reasoning knowledge-update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-strategy", default="rrf")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text())
    if args.types:
        data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    assert args.model != args.judge_model, "a judge must not grade its own answers"
    judge_llm = LLMService(RAGConfig(model=args.judge_model, max_retries=8,
                                     retry_deadline_secs=300))

    print(f"{len(data)} instances | model={args.model} judge={args.judge_model} "
          f"merge={args.merge_strategy}")
    print(f"{'type':<26} {'sess':>5} {'index s':>8} {'query s':>8}  verdict")
    print("-" * 70)

    results, by_type, degraded = [], defaultdict(list), []
    started = time.time()
    for i, inst in enumerate(data):
        realm = f"lme_{args.seed}_{i}"
        rag = GraphRAG(RAGConfig(
            model=args.model, realm=realm, schema_per_realm=True,
            embedding_model=args.embedding_model, embedding_dim=1536,
            embed_relations=True, merge_strategy=args.merge_strategy,
            max_retries=8, retry_deadline_secs=600))
        await rag.initialize()
        try:
            t0 = time.time()
            try:
                n_sessions = await index_instance(rag, inst)
            except DegradedRun as e:
                # Counted separately and never as a wrong answer.
                degraded.append({"question_id": inst["question_id"],
                                 "type": inst["question_type"], "reason": str(e)})
                print(f"{inst['question_type']:<26} {'--':>5} {'--':>8} {'--':>8}  "
                      f"SKIPPED (degraded: {str(e)[:60]})")
                continue
            t_index = time.time() - t0

            t0 = time.time()
            # The question is asked as of a date, and "how many weeks since"
            # is measured from it. Supplying it is not a hint — the benchmark
            # gives it to every system.
            asked = (f"Today is {parse_date(inst.get('question_date', ''))}. "
                     f"{inst['question']}") if inst.get("question_date") else inst["question"]
            out = await rag.query(asked, param=QueryParam(mode="mix", top_k=8))
            t_query = time.time() - t0
            answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()

            ok = await judge(judge_llm, inst["question"], inst["answer"], answer)
            by_type[inst["question_type"]].append(ok)
            results.append({"question_id": inst["question_id"],
                            "type": inst["question_type"], "correct": ok,
                            "index_secs": round(t_index, 1),
                            "query_secs": round(t_query, 1),
                            "question": inst["question"], "gold": inst["answer"],
                            "answer": answer[:1200]})
            print(f"{inst['question_type']:<26} {n_sessions:>5} {t_index:>8.1f} "
                  f"{t_query:>8.1f}  {'CORRECT' if ok else 'incorrect'}")
        finally:
            try:
                await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE;')
            finally:
                await rag.close()

    print("-" * 70)
    total = sum(1 for r in results if r["correct"])
    for qtype, marks in sorted(by_type.items()):
        print(f"  {qtype:<26} {sum(marks):>3}/{len(marks):<3} {100*sum(marks)/len(marks):>5.0f}%")
    print(f"  {'OVERALL':<26} {total:>3}/{len(results):<3} "
          f"{100*total/max(1,len(results)):>5.0f}%")
    if degraded:
        # Loud, and excluded from the denominator: a skipped instance is not a
        # wrong answer, and an accuracy figure computed over a degraded run is
        # not a measurement of this system.
        print(f"\n  !! {len(degraded)} of {len(data)} instances SKIPPED as degraded.")
        print(f"  !! The accuracy above covers {len(results)} instances only, and "
              f"should not be reported until this is zero.")
    print(f"  median query latency: "
          f"{sorted(r['query_secs'] for r in results)[len(results)//2]:.1f}s")
    print(f"  {time.time() - started:.0f}s total")

    pathlib.Path(args.out).write_text(json.dumps(
        {"model": args.model, "judge": args.judge_model,
         "merge_strategy": args.merge_strategy, "n": len(results),
         "accuracy": total / max(1, len(results)),
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded,
         "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
         "results": results}, indent=2))
    print(f"  wrote {args.out}")



if __name__ == "__main__":
    asyncio.run(main())
