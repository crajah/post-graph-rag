#!/usr/bin/env python3
"""Report on an indexed graph: cross-document linkage and extraction quality.

    python evaluation/analyse_graph.py
    python evaluation/analyse_graph.py --realm wiki_kb --ask "Who built the engine?"

Read-only, so it is safe to run while indexing is still in progress.
"""
import argparse
import asyncio
import logging
from collections import defaultdict

from post_graph_rag import GraphRAG, QueryParam, RAGConfig

DEFAULT_QUESTIONS = [
    "What was the relationship between Ada Lovelace and Charles Babbage?",
    "How does the Analytical Engine differ from the Difference Engine?",
]


async def run(args):
    S = f'"{args.realm}"'
    config = RAGConfig(
        model=args.model,
        fallback_models=args.fallback_models,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        realm=args.realm,
        schema_per_realm=True,
    )
    rag = GraphRAG(config)
    await rag.store.connect()
    q = rag.store.client._fetch

    def header(n, title):
        print("\n" + "=" * 72 + f"\n{n}. {title}\n" + "=" * 72)

    header(1, "CORPUS")
    for r in await q(f"SELECT payload->>'document' AS doc, count(*) n FROM {S}.documents GROUP BY 1 ORDER BY 1"):
        print(f"   {str(r['doc'])[:34]:34} {r['n']:3} chunks")
    t = (await q(f"SELECT (SELECT count(*) FROM {S}.documents) d, (SELECT count(*) FROM {S}.entities) e,"
                 f" (SELECT count(*) FROM {S}.relations) r, (SELECT count(*) FROM {S}.doc_mentions) m"))[0]
    print(f"\n   {t['d']} chunks | {t['e']} entities | {t['r']} relations | {t['m']} mentions")

    header(2, "CROSS-DOCUMENT ENTITY RESOLUTION")
    rows = await q(f"""
        SELECT e.payload->>'name' AS name,
               count(DISTINCT d.payload->>'document') AS docs,
               count(DISTINCT m.from_id) AS chunks
        FROM {S}.entities e
        JOIN {S}.doc_mentions m ON m.to_id = e.id
        JOIN {S}.documents d ON d.id = m.from_id
        GROUP BY 1 HAVING count(DISTINCT d.payload->>'document') > 1
        ORDER BY docs DESC, chunks DESC
    """)
    print(f"   Entities appearing in more than one source document: {len(rows)}")
    for r in rows[: args.limit]:
        print(f"     {str(r['name'])[:46]:46} {r['docs']} docs, {r['chunks']} chunks")

    header(3, "CROSS-DOCUMENT TRAVERSAL")
    hubs = await q(f"""
        SELECT e.payload->>'name' AS name, e.id
        FROM {S}.entities e
        JOIN {S}.doc_mentions m ON m.to_id = e.id
        JOIN {S}.documents d ON d.id = m.from_id
        GROUP BY 1,2 HAVING count(DISTINCT d.payload->>'document') > 1
        ORDER BY count(DISTINCT d.payload->>'document') DESC LIMIT 3
    """)
    for h in hubs:
        rels = await q(f"""
            SELECT r.relation_type, t.payload->>'name' AS tgt
            FROM {S}.relations r JOIN {S}.entities t ON t.id = r.to_id
            WHERE r.from_id = {int(h['id'])} LIMIT 8
        """)
        print(f"\n   '{h['name']}':")
        for x in rels:
            print(f"     --[{x['relation_type']}]--> {str(x['tgt'])[:44]}")

    header(4, "PREDICATE VOCABULARY")
    rows = await q(f"SELECT relation_type, count(*) n FROM {S}.relations GROUP BY 1 ORDER BY n DESC")
    total = sum(r["n"] for r in rows)
    once = [r for r in rows if r["n"] == 1]
    print(f"   {len(rows)} distinct predicates over {total} relations "
          f"({len(once)} used exactly once = {100*len(once)//max(1,len(rows))}%)")
    print("   most common:", ", ".join(f"{r['relation_type']}({r['n']})" for r in rows[:12]))

    header(5, "ENTITY TYPE VOCABULARY")
    rows = await q(f"SELECT payload->>'type' AS t, count(*) n FROM {S}.entities GROUP BY 1 ORDER BY n DESC")
    print(f"   {len(rows)} distinct types:", ", ".join(f"{r['t']}({r['n']})" for r in rows[:18]))

    header(6, "LIKELY UNMERGED ALIASES")
    ents = await q(f"SELECT payload->>'name' AS name FROM {S}.entities")
    groups = defaultdict(set)
    for e in ents:
        if not e["name"]:
            continue
        for tok in {w.lower().strip(".,()'\"") for w in e["name"].split() if len(w) > 3}:
            groups[tok].add(e["name"])
    shown = 0
    for tok, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) > 1 and shown < args.limit:
            print(f"   '{tok}': {sorted(members)[:6]}")
            shown += 1

    header(7, "CROSS-DOCUMENT RETRIEVAL")
    for question in (args.ask or DEFAULT_QUESTIONS):
        res = await rag.query_data(question, param=QueryParam(mode=args.mode, top_k=args.top_k))
        docs = sorted({c["metadata"].get("document") for c in res["data"]["chunks"]})
        print(f"\n   Q: {question}")
        print(f"      chunks from: {docs}")
        print(f"      entities:    {[e['entity_name'] for e in res['data']['entities']][:8]}")
        for r in res["data"]["relationships"][:6]:
            print(f"      ({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})")

    await rag.close()


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realm", default=os.getenv("RAG_EVAL_REALM", "wiki_kb"))
    ap.add_argument("--model", default=os.getenv("RAG_MODEL", "MiniMax-M2.7"))
    ap.add_argument("--fallback-models", nargs="*", default=[
        m for m in os.getenv("RAG_FALLBACK_MODELS", "").split(",") if m])
    ap.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    ap.add_argument("--embedding-dim", type=int, default=int(os.getenv("RAG_EMBEDDING_DIM", "1536")))
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20, help="Rows shown per section")
    ap.add_argument("--ask", nargs="*", help="Override the retrieval questions")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
