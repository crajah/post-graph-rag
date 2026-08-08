#!/usr/bin/env python3
"""Compare indexed graphs side by side.

    python evaluation/compare_realms.py wiki_kb3 wiki_kb4 wiki_kb5

Useful for isolating the effect of one variable — a model, a vocabulary, a
chunk size — by indexing the same corpus into different realms and diffing the
resulting graphs.
"""
import argparse
import asyncio
import sys

from post_graph import AsyncPostGraph

try:
    from index_corpus import BIOGRAPHY_VOCABULARY
except ImportError:  # invoked from the repo root
    sys.path.insert(0, "evaluation")
    from index_corpus import BIOGRAPHY_VOCABULARY

METRICS = [
    ("chunks", "documents indexed"),
    ("entities", "entities"),
    ("relations", "relations"),
    ("mentions", "doc_mentions"),
    ("predicates", "distinct predicates"),
    ("single_use", "single-use predicates"),
    ("on_vocab", "relations on vocabulary"),
    ("aliased", "entities carrying aliases"),
    ("negated", "negated relations"),
    ("weighted", "relations corroborated (weight>1)"),
    ("orphans", "orphan entities"),
    ("cross_doc", "entities in >1 document"),
    ("communities", "communities built"),
]


async def collect(client, realm, vocabulary):
    async def q(sql):
        return await client._fetch(sql)

    async def scalar(sql):
        rows = await q(sql)
        return rows[0]["n"] if rows else 0

    preds = await q(f"SELECT relation_type p, count(*) n FROM {realm}.relations GROUP BY 1")
    total = sum(r["n"] for r in preds) or 1
    on_vocab = sum(r["n"] for r in preds if r["p"] in vocabulary)
    once = len([r for r in preds if r["n"] == 1])

    try:
        communities = await scalar(f"SELECT count(*) n FROM {realm}.communities")
    except Exception:
        communities = 0

    return {
        "chunks": await scalar(f"SELECT count(*) n FROM {realm}.documents"),
        "entities": await scalar(f"SELECT count(*) n FROM {realm}.entities"),
        "relations": total,
        "mentions": await scalar(f"SELECT count(*) n FROM {realm}.doc_mentions"),
        "predicates": len(preds),
        "single_use": f"{100 * once // max(1, len(preds))}%",
        "on_vocab": f"{100 * on_vocab // total}%",
        "aliased": await scalar(
            f"SELECT count(*) n FROM {realm}.entities "
            f"WHERE jsonb_array_length(coalesce(payload->'aliases','[]'::jsonb)) > 0"),
        "negated": await scalar(
            f"SELECT count(*) n FROM {realm}.relations WHERE payload->>'negated' = 'true'"),
        "weighted": await scalar(
            f"SELECT count(*) n FROM {realm}.relations WHERE (payload->>'weight')::int > 1"),
        "orphans": await scalar(
            f"SELECT count(*) n FROM {realm}.entities e WHERE NOT EXISTS "
            f"(SELECT 1 FROM {realm}.relations r WHERE r.from_id = e.id OR r.to_id = e.id)"),
        "cross_doc": await scalar(
            f"SELECT count(*) n FROM (SELECT e.id FROM {realm}.entities e "
            f"JOIN {realm}.doc_mentions m ON m.to_id = e.id "
            f"JOIN {realm}.documents d ON d.id = m.from_id GROUP BY e.id "
            f"HAVING count(DISTINCT d.payload->>'document') > 1) x"),
        "communities": communities,
    }


async def run(args):
    client = AsyncPostGraph(dsn=args.dsn)
    await client.connect()

    results = {}
    for realm in args.realms:
        try:
            results[realm] = await collect(client, realm, set(BIOGRAPHY_VOCABULARY))
        except Exception as e:
            print(f"   !! {realm}: {str(e)[:80]}")
    await client.close()

    if not results:
        raise SystemExit("No readable realms.")

    width = max(14, max(len(r) for r in results) + 2)
    print(f"{'metric':36}" + "".join(f"{r:>{width}}" for r in results))
    print("-" * (36 + width * len(results)))
    for key, label in METRICS:
        print(f"{label:36}" + "".join(f"{str(results[r][key]):>{width}}" for r in results))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("realms", nargs="+", help="Realm schema names to compare")
    ap.add_argument("--dsn", default="postgresql://localhost:5432/postgres")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
