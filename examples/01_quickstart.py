"""Index three documents and ask a question that needs all three.

The point: the answer requires joining facts stated in separate documents.
A flat vector store retrieves whichever chunk is most similar to the
question; the graph walks the relations between them.
"""
import asyncio

from _shared import banner, fresh_realm, make_config

from post_graph_rag import GraphRAG

DOCS = [
    ("acquisition.txt",
     "Northwind Systems acquired Vertex Analytics in March 2021. "
     "Vertex Analytics was founded by Dr. Amara Okafor in 2014."),
    ("leadership.txt",
     "Dr. Amara Okafor was appointed Chief Technology Officer of Northwind "
     "Systems in June 2021, reporting to CEO Martin Feld."),
    ("products.txt",
     "Northwind's flagship product, Meridian, was rebuilt in 2022 on the "
     "forecasting engine that Vertex Analytics originally developed."),
]


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_quickstart")))
    await rag.initialize()
    try:
        for source, text in DOCS:
            res = await rag.index_document(text, metadata={"source": source})
            print(f"indexed {source}: {res['entities_extracted']} entities")

        banner("Who built the engine behind Meridian, and where do they work now?")
        out = await rag.query(
            "Who built the technology behind Meridian, and what is their role "
            "at Northwind today?")
        print(out["answer"])

        banner("Relations the answer was built from")
        for t in out.get("retrieved_graph_triples", [])[:8]:
            print(f"  {t}")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
