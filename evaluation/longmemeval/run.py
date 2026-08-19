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
import logging
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).resolve().parent

# The library's default extraction prompt forbids exactly what a chat log is
# made of. It says: never emit a pronoun as an entity, never emit a possessive
# role phrase, emit only the stable entity that could be named again in a
# different document. In a conversation the central entity is the speaker —
# "I", "my" — and the facts are possessive: my sneakers, my wake-up time, my
# loyalty tier. Those rules are right for filings and wrong here, and the first
# run's failures were all of that shape.
CONVERSATIONAL_PROMPT = """You are extracting a knowledge graph from a chat log
between a user and an assistant.

Extract ENTITIES and TRIPLES that capture what is true about the USER.

The speaker is an entity. Emit the user as the entity 'User'. First-person
pronouns — I, me, my, mine — all refer to 'User'. This is the opposite of the
usual rule against pronoun entities, and it is deliberate: in a conversation the
speaker is the subject of almost every fact worth keeping.

KEEP, as triples from 'User':
- Possessions and their locations: (User, keeps_sneakers_in, shoe rack in closet)
- Habits, routines and times: (User, wakes_up_at, 6:45 AM)
- Preferences and dislikes: (User, prefers, oat milk)
- Status, memberships, tiers: (User, has_status, Premier Silver)
- Purchases and quantities: (User, owns_count_of_tops, five)
- Events the user took part in, with WHEN: (User, attended, Summer Nights festival)
- Plans, decisions and changes of mind

DATES ARE FACTS, NOT DECORATION.
Each conversation begins with a line '[Conversation on YYYY-MM-DD]'. That is
when the statements in it were made. Put it in valid_from on EVERY triple you
extract from that conversation. A question like "how many weeks ago did I attend
the festival" is unanswerable without it, and it is the single most common
reason an answer cannot be produced.

Where the text states its own date ("last Tuesday", "in March"), resolve it
against the conversation date and use the resolved date.

DO NOT extract:
- Assistant suggestions the user did not adopt.
- Generic advice, or abstractions like 'Product variety' or 'Seasonal flavors'.
  A concept nobody could ask a question about is not worth a vertex.

Predicates should be specific and readable: 'keeps_in', 'wakes_up_at',
'attended', 'purchased', 'has_status', 'prefers', 'lives_in', 'works_as'.

Return strictly the required JSON structure."""

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


async def judge_panel(llms, question, gold, answer):
    """Majority vote over a panel. Returns (verdict, per-judge votes).

    One judge is one model's opinion, and this repository has repeatedly found
    model choice dominating results — including on the judging side, where
    re-grading the same stored answers moved a score by five points. A panel
    does not remove that, but it stops a single grader's bias deciding the
    number on its own.

    A judge that fails or replies unusably does not vote. The majority is taken
    over votes actually cast, and the ballot is recorded so a lopsided panel is
    visible rather than averaged away.
    """
    votes = {}
    for name, llm in llms.items():
        try:
            votes[name] = await judge(llm, question, gold, answer)
        except Exception:
            logger.warning("judge %s failed to vote", name)
    if not votes:
        # No verdict is not a pass.
        return False, votes
    return sum(votes.values()) * 2 > len(votes), votes


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
    ap.add_argument("--judge-model", default="gemini-3.6-flash",
                    help="single judge; ignored when --judges is given")
    ap.add_argument("--judges", nargs="*", default=None,
                    help="panel of judges; a majority marks an answer correct")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--types", nargs="*", default=None,
                    help="question_type filter, e.g. temporal-reasoning knowledge-update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-strategy", default="rrf")
    ap.add_argument("--conversational", dest="conversational", action="store_true",
                    default=True, help="use the chat-tuned extraction prompt")
    ap.add_argument("--no-conversational", dest="conversational", action="store_false")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="instances indexed and queried at once")
    ap.add_argument("--chunk-concurrency", type=int, default=8,
                    help="chunks extracted at once within one instance")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    # Ablation switches. Each of the three features taken from Graphiti is off
    # by default, so a run with no switches reproduces the 75% baseline and any
    # difference is attributable to the one flag that was turned on.
    ap.add_argument("--mmr", action="store_true",
                    help="diversify the merged candidates by maximal marginal relevance")
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    ap.add_argument("--node-distance", action="store_true",
                    help="rerank by graph distance from the matched entities")
    ap.add_argument("--contradiction", action="store_true",
                    help="ask the model which existing facts a new one retracts")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text())
    if args.types:
        data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    panel_names = args.judges or [args.judge_model]
    if args.model in panel_names:
        raise SystemExit(f"{args.model} cannot both answer and judge")
    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900))
                  for n in panel_names}

    # Preflight: index one real session and confirm relations come back before
    # committing to the full sweep. Four runs have died on model availability —
    # a name the router advertises but does not serve, a name differing by one
    # character — and each cost an hour to discover what this costs seconds.
    probe = GraphRAG(RAGConfig(
        model=args.model, realm="lme_preflight", schema_per_realm=True,
        embedding_model=args.embedding_model, embedding_dim=1536,
        embed_relations=True, max_retries=6, retry_deadline_secs=180,
        extraction_prompt=CONVERSATIONAL_PROMPT if args.conversational else None))
    await probe.initialize()
    try:
        sample = data[0]
        await probe.index_document(
            render_session(sample["haystack_sessions"][0],
                           parse_date(sample["haystack_dates"][0])),
            metadata=DocumentMetadata(document="preflight"))
        rels = await probe.store.get_all_relations(limit=5)
        if not rels:
            raise SystemExit(
                f"preflight failed: {args.model} indexed a session and produced no "
                f"relations. The sweep would report every instance as degraded.")
        print(f"preflight OK: {args.model} produced {len(rels)} relations")
    finally:
        try:
            await probe.store.client._execute('DROP SCHEMA IF EXISTS "lme_preflight" CASCADE;')
        finally:
            await probe.close()

    print(f"{len(data)} instances | model={args.model} judge={args.judge_model} "
          f"merge={args.merge_strategy}")
    on = [n for n, v in (("mmr", args.mmr), ("node-distance", args.node_distance),
                         ("contradiction", args.contradiction)) if v]
    print(f"features: {', '.join(on) if on else 'none (baseline)'}")
    print(f"judges: {', '.join(panel_names)}")
    print(f"{'type':<26} {'sess':>5} {'index s':>8} {'query s':>8}  verdict")
    print("-" * 70)

    results, by_type, degraded = [], defaultdict(list), []
    started = time.time()
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(i, inst):
        """Index and answer a single instance in its own realm.

        Instances are independent haystacks, so they parallelise cleanly: the
        only shared resource is the model router, which the semaphore bounds.
        """
        nonlocal done
        async with sem:
            realm = f"lme_{args.seed}_{i}"
            rag = GraphRAG(RAGConfig(
                model=args.model, realm=realm, schema_per_realm=True,
                embedding_model=args.embedding_model, embedding_dim=1536,
                embed_relations=True, merge_strategy=args.merge_strategy,
                max_concurrent_chunks=args.chunk_concurrency,
                extraction_prompt=CONVERSATIONAL_PROMPT if args.conversational else None,
                mmr_enabled=args.mmr, mmr_lambda=args.mmr_lambda,
                node_distance_rerank=args.node_distance,
                contradiction_detection=args.contradiction,
                max_retries=40, retry_deadline_secs=1800))
            await rag.initialize()
            try:
                t0 = time.time()
                try:
                    n_sessions = await index_instance(rag, inst)
                except DegradedRun as e:
                    degraded.append({"question_id": inst["question_id"],
                                     "type": inst["question_type"], "reason": str(e)})
                    print(f"{inst['question_type']:<26} {'--':>5} {'--':>8} {'--':>8}  "
                          f"SKIPPED ({str(e)[:50]})", flush=True)
                    return
                t_index = time.time() - t0

                t0 = time.time()
                asked = (f"Today is {parse_date(inst.get('question_date', ''))}. "
                         f"{inst['question']}") if inst.get("question_date") else inst["question"]
                out = await rag.query(asked, param=QueryParam(mode="mix", top_k=8))
                t_query = time.time() - t0
                answer = (out["answer"] if isinstance(out, dict) else str(out)).strip()

                ok, votes = await judge_panel(judge_llms, inst["question"],
                                              inst["answer"], answer)
                by_type[inst["question_type"]].append(ok)
                results.append({"question_id": inst["question_id"],
                                "type": inst["question_type"], "correct": ok,
                                "index_secs": round(t_index, 1),
                                "query_secs": round(t_query, 1),
                                "question": inst["question"], "gold": inst["answer"],
                                "votes": votes, "answer": answer[:1200]})
                done += 1
                print(f"{inst['question_type']:<26} {n_sessions:>5} {t_index:>8.1f} "
                      f"{t_query:>8.1f}  {'CORRECT' if ok else 'incorrect':<9} "
                      f"[{done}/{len(data)}]", flush=True)
            except Exception as e:
                degraded.append({"question_id": inst["question_id"],
                                 "type": inst["question_type"],
                                 "reason": f"{type(e).__name__}: {e}"})
                print(f"{inst['question_type']:<26} {'--':>5} {'--':>8} {'--':>8}  "
                      f"FAILED ({type(e).__name__})", flush=True)
            finally:
                try:
                    await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE;')
                finally:
                    await rag.close()

    await asyncio.gather(*(one(i, inst) for i, inst in enumerate(data)))

    print("-" * 70)
    total = sum(1 for r in results if r["correct"])
    for qtype, marks in sorted(by_type.items()):
        print(f"  {qtype:<26} {sum(marks):>3}/{len(marks):<3} {100*sum(marks)/len(marks):>5.0f}%")
    print(f"  {'OVERALL':<26} {total:>3}/{len(results):<3} "
          f"{100*total/max(1,len(results)):>5.0f}%")
    if results and len(panel_names) > 1:
        print("\n  per-judge agreement with the panel verdict:")
        for name in panel_names:
            cast = [(r["correct"], r["votes"][name]) for r in results if name in r.get("votes", {})]
            if not cast:
                print(f"    {name:<24} cast no votes")
                continue
            agree = sum(1 for v, own in cast if v == own)
            print(f"    {name:<24} {agree}/{len(cast)} ({100*agree/len(cast):.0f}%)")

    if degraded:
        # Loud, and excluded from the denominator: a skipped instance is not a
        # wrong answer, and an accuracy figure computed over a degraded run is
        # not a measurement of this system.
        print(f"\n  !! {len(degraded)} of {len(data)} instances SKIPPED as degraded.")
        print(f"  !! The accuracy above covers {len(results)} instances only, and "
              f"should not be reported until this is zero.")
    if results:
        print(f"  median query latency: "
              f"{sorted(r['query_secs'] for r in results)[len(results)//2]:.1f}s")
    else:
        print("  no instance completed; see the degraded reasons above")
    print(f"  {time.time() - started:.0f}s total")

    pathlib.Path(args.out).write_text(json.dumps(
        {"model": args.model, "judge": panel_names,
         "merge_strategy": args.merge_strategy, "n": len(results),
         # Self-describing: an ablation result is meaningless without the
         # switches it was taken under, and these files outlive the shell
         # history that produced them.
         "features": {"mmr": args.mmr, "mmr_lambda": args.mmr_lambda,
                      "node_distance": args.node_distance,
                      "contradiction": args.contradiction},
         "accuracy": total / max(1, len(results)),
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded,
         "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
         "results": results}, indent=2))
    print(f"  wrote {args.out}")



if __name__ == "__main__":
    asyncio.run(main())
