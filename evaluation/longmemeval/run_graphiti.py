"""LongMemEval against Graphiti, on the same PostgreSQL post-graph stores.

The point of the comparison is the engine, not the storage, so as much as
possible is held identical to `run.py`: the same instances, the same answering
model, the same judge panel, the same scoring function, and the same
per-instance isolation. Graphiti runs on its post-graph driver
(getzep/graphiti#1777), so both systems read and write the same database.

What necessarily differs is the ingestion unit. post-graph-rag indexes a
session as a document; Graphiti ingests it as an *episode*, which is its own
first-class concept. Using each system's native unit is the fair comparison —
forcing one into the other's shape would measure the adapter.
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

API_BASE = __import__("os").getenv("OPENAI_API_BASE", "http://localhost:4010/v1")
API_KEY = __import__("os").getenv("OPENAI_API_KEY", "")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "graphiti"))

from run import judge_panel, parse_date, render_session  # noqa: E402

from post_graph_rag import RAGConfig  # noqa: E402
from post_graph_rag.llm import LLMService  # noqa: E402


async def answer_from_context(llm, question: str, context: str) -> str:
    """Synthesise an answer from retrieved context.

    Deliberately the same shape of prompt for both systems. Graphiti's search
    returns facts rather than an answer, so without this the comparison would
    be Graphiti's retrieval against post-graph-rag's retrieval *and*
    synthesis — and the difference would be unattributable.
    """
    prompt = (f"Answer the question using only the facts below, retrieved from a "
              f"memory of the user's chat history. If they do not support an "
              f"answer, say so plainly.\n\n---Facts---\n{context}\n\n"
              f"---Question---\n{question}\n\nAnswer concisely.")
    reply = await llm.chat_completion([{"role": "user", "content": prompt}])
    return (reply if isinstance(reply, str) else str(reply)).strip()


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
    ap.add_argument("--dsn", default="postgresql://localhost:5432/postgres")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results_graphiti.json"))
    args = ap.parse_args()

    from graphiti_core import Graphiti
    # The package must be imported before the module: postgraph/__init__.py
    # imports from postgraph_driver, which imports back into the package, and
    # importing the module directly hits the half-built cycle. The driver's own
    # tests happen to take the working order, so the bug is invisible to them.
    import graphiti_core.driver.postgraph  # noqa: F401
    from graphiti_core.driver.postgraph_driver import PostGraphDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.nodes import EpisodeType

    data = json.loads(pathlib.Path(args.data).read_text())
    data = [d for d in data if d["question_type"] in args.types]
    random.Random(args.seed).shuffle(data)
    data = data[:args.limit]

    judge_llms = {n: LLMService(RAGConfig(model=n, max_retries=25,
                                          retry_deadline_secs=900))
                  for n in args.judges}
    answer_llm = LLMService(RAGConfig(model=args.model, max_retries=25,
                                      retry_deadline_secs=900))

    print(f"{len(data)} instances | graphiti on post-graph | model={args.model}")
    print(f"judges: {', '.join(args.judges)}")
    print(f"{'type':<26} {'sess':>5} {'index s':>8} {'query s':>8}  verdict")
    print("-" * 70)

    results, by_type, degraded = [], defaultdict(list), []
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()

    async def one(i, inst):
        async with sem:
            group_id = f"lmeg_{args.seed}_{i}"
            driver = PostGraphDriver(dsn=args.dsn, embedding_dim=1536)
            # Graphiti builds its own OpenAI clients and would otherwise call
            # api.openai.com with the router key. Both systems must use the same
            # models through the same endpoint or the comparison is meaningless.
            llm_cfg = LLMConfig(api_key=API_KEY, base_url=API_BASE,
                                model=args.model, small_model=args.model)
            client = Graphiti(
                graph_driver=driver,
                llm_client=OpenAIClient(config=llm_cfg),
                embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(
                    api_key=API_KEY, base_url=API_BASE,
                    embedding_model=args.embedding_model, embedding_dim=1536)))
            try:
                await client.build_indices_and_constraints()
                sessions = sorted(
                    zip(inst["haystack_dates"], inst["haystack_session_ids"],
                        inst["haystack_sessions"]),
                    key=lambda s: parse_date(s[0]) or s[0])

                t0 = time.time()
                for raw_date, session_id, turns in sessions:
                    body = render_session(turns, parse_date(raw_date))
                    if not body.strip():
                        continue
                    # reference_time is Graphiti's event-time axis, the
                    # counterpart of valid_from here. Supplying the session date
                    # gives it the same information run.py writes into the text.
                    await client.add_episode(
                        name=session_id, episode_body=body,
                        source=EpisodeType.message,
                        source_description="chat session",
                        reference_time=_as_datetime(parse_date(raw_date)),
                        group_id=group_id)
                t_index = time.time() - t0

                t0 = time.time()
                asked = (f"Today is {parse_date(inst.get('question_date', ''))}. "
                         f"{inst['question']}") if inst.get("question_date") else inst["question"]
                hits = await client.search(asked, group_ids=[group_id],
                                           num_results=args.top_k)
                context = "\n".join(f"[{n+1}] {h.fact}" for n, h in enumerate(hits))
                answer = await answer_from_context(answer_llm, asked, context)
                t_query = time.time() - t0

                ok, votes = await judge_panel(judge_llms, inst["question"],
                                              inst["answer"], answer)
                by_type[inst["question_type"]].append(ok)
                results.append({"question_id": inst["question_id"],
                                "type": inst["question_type"], "correct": ok,
                                "index_secs": round(t_index, 1),
                                "query_secs": round(t_query, 1), "facts": len(hits),
                                "question": inst["question"], "gold": inst["answer"],
                                "votes": votes, "answer": answer[:1200]})
                print(f"{inst['question_type']:<26} {len(sessions):>5} {t_index:>8.1f} "
                      f"{t_query:>8.1f}  {'CORRECT' if ok else 'incorrect'}", flush=True)
            except Exception as e:
                degraded.append({"question_id": inst["question_id"],
                                 "type": inst["question_type"],
                                 "reason": f"{type(e).__name__}: {e}"})
                import traceback
                traceback.print_exc()
                print(f"{inst['question_type']:<26} {'--':>5} {'--':>8} {'--':>8}  "
                      f"FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
            finally:
                await driver.close()

    await asyncio.gather(*(one(i, inst) for i, inst in enumerate(data)))

    print("-" * 70)
    total = sum(1 for r in results if r["correct"])
    for qtype, marks in sorted(by_type.items()):
        print(f"  {qtype:<26} {sum(marks):>3}/{len(marks):<3} "
              f"{100*sum(marks)/len(marks):>5.0f}%")
    print(f"  {'OVERALL':<26} {total:>3}/{len(results):<3} "
          f"{100*total/max(1,len(results)):>5.0f}%")
    if degraded:
        print(f"\n  !! {len(degraded)} of {len(data)} instances FAILED — not reportable")
        for d in degraded[:3]:
            print(f"     {d['reason'][:100]}")
    if results:
        print(f"  median query latency: "
              f"{sorted(r['query_secs'] for r in results)[len(results)//2]:.1f}s")
    print(f"  {time.time() - started:.0f}s total")

    pathlib.Path(args.out).write_text(json.dumps(
        {"system": "graphiti-on-postgraph", "model": args.model,
         "judge": args.judges, "n": len(results),
         "accuracy": total / max(1, len(results)),
         "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded, "results": results}, indent=2))
    print(f"  wrote {args.out}")


def _as_datetime(iso_date: str):
    from datetime import datetime, timezone
    if not iso_date:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    asyncio.run(main())
