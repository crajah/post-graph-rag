"""Corpus-level questions, and deciding what to look at next.

build_communities clusters the graph and writes an LLM report per cluster, so
"what are the themes here" is answered from summaries rather than from a
top-k of chunks. With community_levels > 1 the clusters nest, giving you a
topic tree to drill through.

The exploration calls answer a different question -- not "what is the answer"
but "where has nobody looked". Coverage telemetry is opt-in and stores only a
hash of each query.
"""
import asyncio

from _shared import banner, fresh_realm, make_config
from post_graph_rag import GraphRAG, QueryParam

DOCS = [
    "Solar capacity in Spain grew 22 percent as grid connections accelerated.",
    "Portugal commissioned three offshore wind farms off the Atlantic coast.",
    "Battery storage costs fell sharply, improving solar project economics.",
    "Rail freight volumes in Germany declined amid industrial slowdown.",
    "Dutch port throughput fell as container traffic shifted south.",
    "Rail operators in France invested in electrification of regional lines.",
]


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_communities"),
                               community_levels=2,
                               record_retrieval_events=True))
    await rag.initialize()
    try:
        for i, text in enumerate(DOCS):
            await rag.index_document(text, metadata={"source": f"news_{i}.txt"})
        print(f"indexed {len(DOCS)} documents")

        res = await rag.build_communities()
        print(f"communities: {res['communities']} across levels {res['levels']}")

        banner("A corpus-level question, answered from cluster reports")
        out = await rag.query("What are the main themes across this corpus?",
                              param=QueryParam(mode="global"))
        print(out["answer"])

        banner("The topic tree")
        tree = await rag.get_community_tree()
        for root in tree["roots"]:
            print(f"  L{root['level']}: {root.get('title')}")
            for child in root.get("children", []):
                print(f"    L{child['level']}: {child.get('title')}")

        banner("Where has retrieval never looked?")
        frontier = await rag.least_explored_communities(k=3)
        for c in frontier:
            print(f"  {c.title}  ({c.members} entities, "
                  f"{c.retrieval_hits} retrieval hits)")
        dark = await rag.dark_entities(limit=5)
        print(f"\n  never-retrieved entities: {len(dark)}")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
