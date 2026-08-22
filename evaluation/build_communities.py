#!/usr/bin/env python3
"""Cluster an indexed graph, summarise each community, and answer a corpus-level question.

    python evaluation/build_communities.py --realm wiki_kb3
    python evaluation/build_communities.py --ask "What themes connect these documents?"

Run after index_corpus.py. Corpus-level questions cannot be answered by retrieving
passages, because no single passage contains the answer.
"""
import argparse
import asyncio
import logging
import os
import time

from post_graph_rag import GraphRAG, QueryParam, RAGConfig

DEFAULT_QUESTIONS = [
    "What are the main themes across these documents?",
    "How are the people and the machines in this corpus connected?",
]


async def run(args):
    config = RAGConfig(
        model=args.model,
        fallback_models=args.fallback_models,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        realm=args.realm,
        schema_per_realm=True,
        community_min_size=args.min_size,
        community_resolution=args.resolution,
        max_communities=args.max_communities,
    )
    rag = GraphRAG(config)
    await rag.store.connect()
    await rag.store.initialize_schema()

    print("=" * 72)
    print(f"BUILDING COMMUNITIES (realm={args.realm})")
    print("=" * 72)
    t0 = time.time()
    res = await rag.build_communities()
    print(f"   detected {res.get('detected', 0)} communities over {res['entities']} entities "
          f"/ {res.get('relations', 0)} relations")
    print(f"   summarised {res['communities']}, skipped {res['skipped']}  ({time.time()-t0:.1f}s)")

    rows = await rag.store.client._fetch(
        f'SELECT payload FROM "{args.realm}"."communities" ORDER BY id'
    )
    print("\n" + "=" * 72)
    print("COMMUNITY REPORTS")
    print("=" * 72)
    import json as _json
    for r in rows:
        p = r["payload"] if isinstance(r["payload"], dict) else _json.loads(r["payload"])
        print(f"\n   [{p.get('size')} entities, importance {p.get('rating')}] {p.get('title')}")
        print(f"   {(p.get('summary') or '')[:300]}")
        for f in (p.get("findings") or [])[:3]:
            print(f"     - {f.get('summary')}")

    print("\n" + "=" * 72)
    print("CORPUS-LEVEL RETRIEVAL (global mode)")
    print("=" * 72)
    for question in (args.ask or DEFAULT_QUESTIONS):
        data = await rag.query_data(question, param=QueryParam(mode="global", top_k=args.top_k))
        print(f"\n   Q: {question}")
        print(f"      communities used: {[c['title'] for c in data['data']['communities']]}")
        if args.synthesise:
            out = await rag.query(question, param=QueryParam(mode="global", top_k=args.top_k))
            print(f"      ANSWER: {out['answer'][:700]}")

    await rag.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realm", default=os.getenv("RAG_EVAL_REALM", "wiki_kb"))
    ap.add_argument("--model", default=os.getenv("RAG_MODEL", "gemini-3.6-flash"))
    ap.add_argument("--fallback-models", nargs="*", default=[
        m for m in os.getenv("RAG_FALLBACK_MODELS", "").split(",") if m])
    ap.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001"))
    ap.add_argument("--embedding-dim", type=int, default=int(os.getenv("RAG_EMBEDDING_DIM", "1536")))
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--max-communities", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--ask", nargs="*")
    ap.add_argument("--synthesise", action="store_true", help="Also call the LLM for a final answer")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
