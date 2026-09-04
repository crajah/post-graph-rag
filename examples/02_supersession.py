"""A later document retires an earlier fact.

This is the behaviour the library exists for. Three filings arrive in order;
each names a different CFO. A store that only accumulates will answer with
all three, or with whichever chunk embeds closest. Here the last assertion
wins and the earlier ones are marked superseded -- kept, not deleted.
"""
import asyncio

from _shared import banner, fresh_realm, make_config
from post_graph_rag import GraphRAG, QueryParam

FILINGS = [
    ("2019-annual-report.txt",
     "Helena Voss serves as Chief Financial Officer of Calder Industries."),
    ("2021-annual-report.txt",
     "Calder Industries announced that Ravi Chandran has been appointed "
     "Chief Financial Officer, succeeding Helena Voss."),
    ("2023-annual-report.txt",
     "Ravi Chandran stepped down as Chief Financial Officer of Calder "
     "Industries. Priya Nair now serves as Chief Financial Officer."),
]


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_supersession")))
    await rag.initialize()
    try:
        for source, text in FILINGS:
            await rag.index_document(text, metadata={"source": source})
            print(f"indexed {source}")

        banner("Who is the CFO of Calder Industries?")
        out = await rag.query("Who is the Chief Financial Officer of Calder Industries?")
        print(out["answer"])
        print("\n-> Priya Nair is named as the current CFO. The earlier two are\n"
              "   reported as prior holders, not as competing present-tense facts --\n"
              "   because the graph closed them when the later filing arrived.")

        banner("Same question, superseded history included")
        out = await rag.query(
            "Who has served as Chief Financial Officer of Calder Industries, "
            "and in what order?",
            param=QueryParam(include_superseded=True))
        print(out["answer"])
        print("\n-> The history is retained and auditable, not overwritten.")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
