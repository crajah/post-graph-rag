"""TEMPO: temporal reasoning-intensive retrieval (arXiv:2601.09523).

Unlike LongMemEval and ECT-QA, TEMPO scores *retrieval*, not answers: given a
query, return documents, and be judged on whether the gold ones come back and
whether the required time periods are covered.

## The reduced pool, and why

TEMPO is 1.65M documents across 13 domains. The `quant` domain alone is 28,785
documents and 59.8M characters. This library indexes by LLM extraction, at a
measured ~2.1M characters per hour on this hardware — so one domain is roughly
thirty hours, and the full benchmark a month. That is not a defect being hidden:
an extraction-based index costs orders of magnitude more per document than the
embedding-only systems TEMPO was built to rank, and the benchmark's own leaders
are dense retrievers.

So this indexes a **pool**: every gold document for the sampled queries, plus
random distractors from the same domain to `--pool-size`. Retrieval over a
thousand documents is easier than over 28,785, and the numbers here are NOT
comparable to the leaderboard. What they can support is a comparison between
configurations of this library on identical pools, which is what the ablation
switches are for.

## Metrics

nDCG@10 and Recall@10 are computed exactly, from `gold_ids`.

`anchor_coverage@10` is NOT the paper's Temporal Coverage@k. TC@k uses an
LLM judge for period coverage; this counts how many of the query's stated
`key_time_anchors` appear in the retrieved text. It is a cheap deterministic
proxy, named differently on purpose so the two are never conflated.
"""
import argparse
import asyncio
import json
import math
import pathlib
import random
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig  # noqa: E402


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"\s+", " ", text).strip()


def ndcg_at_k(ranked_ids, gold, k=10):
    """Binary-relevance nDCG. Gold documents are unordered, so the ideal
    ranking is however many of them exist, packed into the top positions."""
    gains = [1.0 if d in gold else 0.0 for d in ranked_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def anchor_coverage(passages, anchors):
    """Share of the query's stated time anchors that appear in the retrieved text.

    A proxy for the paper's Temporal Coverage@k, not the thing itself — see the
    module docstring. Anchors are short strings like '2019' or 'T-1'.
    """
    if not anchors:
        return None
    blob = " ".join(passages).lower()
    return sum(1 for a in anchors if str(a).lower() in blob) / len(anchors)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quant")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--fallback-models", nargs="*",
                    default=["gemini-3.5-flash-lite", "Meta-Llama-3.3-70B-Instruct"])
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--pool-size", type=int, default=800,
                    help="documents indexed: all gold docs, then distractors to this many")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realm", default="tempo")
    ap.add_argument("--index-concurrency", type=int, default=8)
    ap.add_argument("--query-concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    import pandas as pd
    data = pathlib.Path(args.data)
    steps = pd.read_parquet(data / f"steps_{args.domain}.parquet")
    docs = pd.read_parquet(data / f"docs_{args.domain}.parquet")

    rng = random.Random(args.seed)
    rows = steps.to_dict("records")
    rng.shuffle(rows)
    rows = rows[:args.queries]

    gold_needed = {g for r in rows for g in list(r["gold_ids"])}
    by_id = dict(zip(docs["id"], docs["content"]))
    # A gold document the corpus does not contain would make its query
    # unanswerable for a reason that has nothing to do with retrieval.
    missing = {g for g in gold_needed if g not in by_id}
    if missing:
        print(f"warning: {len(missing)} gold documents absent from the corpus", flush=True)
    gold_present = sorted(gold_needed & set(by_id))

    distractor_pool = [d for d in by_id if d not in gold_needed]
    rng.shuffle(distractor_pool)
    pool = gold_present + distractor_pool[:max(0, args.pool_size - len(gold_present))]

    print(f"TEMPO {args.domain} | {len(rows)} queries | pool {len(pool)} docs "
          f"({len(gold_present)} gold + {len(pool)-len(gold_present)} distractors) "
          f"of {len(by_id)} in domain")
    print(f"model={args.model} realm={args.realm}\n")

    rag = GraphRAG(RAGConfig(
        model=args.model, realm=args.realm, schema_per_realm=True,
        embedding_model=args.embedding_model, embedding_dim=1536,
        embed_relations=True, merge_strategy="rrf",
        fallback_models=args.fallback_models,
        max_retries=args.max_retries, retry_deadline_secs=1800))
    await rag.initialize()

    started = time.time()
    degraded = []
    try:
        if not args.skip_index:
            isem = asyncio.Semaphore(args.index_concurrency)
            done = [0]

            async def index_doc(doc_id):
                async with isem:
                    try:
                        await rag.index_text(
                            strip_html(by_id[doc_id]),
                            metadata=DocumentMetadata(document=doc_id, source="tempo"))
                    except Exception as e:
                        degraded.append({"doc": doc_id, "reason": f"{type(e).__name__}: {e}"})
                    done[0] += 1
                    if done[0] % 50 == 0:
                        print(f"  indexed {done[0]}/{len(pool)} "
                              f"({time.time()-started:.0f}s)", flush=True)

            await asyncio.gather(*(index_doc(d) for d in pool))
            print(f"indexing done in {time.time()-started:.0f}s, "
                  f"{len(degraded)} failures\n", flush=True)

        results = []
        qsem = asyncio.Semaphore(args.query_concurrency)

        async def one(row):
            async with qsem:
                query = strip_html(row["query"])[:1500]
                gold = set(row["gold_ids"])
                guidance = row.get("query_guidance") or {}
                anchors = list(guidance.get("key_time_anchors") or [])
                t0 = time.time()
                try:
                    out = await rag.retrieve(query, param=QueryParam(
                        mode="mix", top_k=args.top_k))
                except Exception as e:
                    degraded.append({"query": row["id"], "reason": f"{type(e).__name__}: {e}"})
                    print(f"  {row['id']:<16} QUERY FAILED {type(e).__name__}", flush=True)
                    return
                chunks = (out.get("data") or {}).get("chunks") or []
                # Chunks carry the document id they came from; several chunks of
                # one document count once, in the order the ranking first saw it.
                ranked, seen = [], set()
                for c in chunks:
                    doc_id = (c.get("metadata") or {}).get("document")
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        ranked.append(doc_id)
                passages = [str(c.get("content", "")) for c in chunks[:args.top_k]]

                hit = len(set(ranked[:args.top_k]) & gold)
                results.append({
                    "id": row["id"], "gold": len(gold), "hits": hit,
                    "ndcg@10": ndcg_at_k(ranked, gold, args.top_k),
                    "recall@10": hit / len(gold) if gold else 0.0,
                    "anchor_coverage@10": anchor_coverage(passages, anchors),
                    "anchors": [str(a) for a in anchors],
                    "retrieved": ranked[:args.top_k],
                    "secs": round(time.time() - t0, 1)})
                print(f"  {row['id']:<16} nDCG={results[-1]['ndcg@10']:.3f} "
                      f"recall={results[-1]['recall@10']:.2f} "
                      f"hits={hit}/{len(gold)}  [{len(results)}/{len(rows)}]", flush=True)

        await asyncio.gather(*(one(r) for r in rows))
    finally:
        await rag.close()

    def mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "-" * 58)
    print(f"  queries scored          {len(results)}")
    print(f"  nDCG@10                 {mean('ndcg@10'):.3f}")
    print(f"  Recall@10               {mean('recall@10'):.3f}")
    print(f"  anchor_coverage@10      {mean('anchor_coverage@10'):.3f}  (proxy, not TC@k)")
    print(f"  models served           {dict(rag.llm.served)}")
    if degraded:
        print(f"\n  !! {len(degraded)} failures — not reportable")

    pathlib.Path(args.out).write_text(json.dumps(
        {"benchmark": "TEMPO", "variant": "reduced-pool", "domain": args.domain,
         "model": args.model, "served": dict(rag.llm.served),
         "pool_size": len(pool), "domain_size": len(by_id),
         "queries": len(results),
         "ndcg@10": mean("ndcg@10"), "recall@10": mean("recall@10"),
         "anchor_coverage@10": mean("anchor_coverage@10"),
         "degraded": degraded, "degraded_count": len(degraded),
         "reportable": not degraded, "results": results}, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
