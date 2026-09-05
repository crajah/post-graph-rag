"""Re-index changed documents, and poll for what moved.

Two properties worth knowing:

  * Re-indexing an unchanged document is a no-op -- same content, no new
    chunks, no duplicated entities, and an empty delta.
  * changes_since(watermark) reports what actually moved from belief time:
    new and superseded relations, new and dormant entities, new documents.
    The default summary transfers counts only, so a poller pays one cheap
    round trip per tick.

Watermarks come from the database clock, so the poll chain stays
exactly-once even when your application server's clock drifts.
"""
import asyncio

from _shared import banner, fresh_realm, make_config

from post_graph_rag import GraphRAG


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_deltas")))
    await rag.initialize()
    try:
        await rag.index_document(
            "Orion Freight operates 40 vehicles from a depot in Leeds.",
            metadata={"source": "fleet.txt"})

        watermark = await rag.watermark()          # start of the poll chain
        print(f"initial watermark: {watermark}")

        banner("Re-index the same content unchanged")
        res = await rag.index_document(
            "Orion Freight operates 40 vehicles from a depot in Leeds.",
            metadata={"source": "fleet.txt"})
        print(f"result: {res}")
        delta = await rag.changes_since(watermark)
        print(f"delta empty: {delta.empty}   -> idempotent re-indexing")

        banner("Now the fact actually changes")
        await rag.index_document(
            "Orion Freight has expanded to 65 vehicles and opened a second "
            "depot in Manchester.",
            metadata={"source": "fleet.txt"})
        delta = await rag.changes_since(watermark)
        print(f"delta empty: {delta.empty}")
        print(f"counts: {delta}")

        detail = await rag.changes_since(watermark, summary=False)
        print(f"\nsuperseded relations: {len(detail.superseded_relations)}")
        print(f"new relations:        {len(detail.new_relations)}")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
