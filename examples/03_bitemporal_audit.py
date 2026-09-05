"""Two clocks: what was true, and what we believed.

A filing published in 2024 can restate a figure for 2019. Those are different
questions and they need different answers:

    as_of            -- the world as it was at that date (valid time)
    as_believed_at   -- the graph as this system held it at that instant
                        (belief time)

The second is the one an auditor asks: not "what is true" but "what did your
systems report in March, before the restatement". Answering it is how you
reproduce a past decision.
"""
import asyncio
from datetime import datetime, timezone

from _shared import banner, fresh_realm, make_config

from post_graph_rag import GraphRAG, QueryParam


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_bitemporal")))
    await rag.initialize()
    try:
        await rag.index_document(
            "In its 2019 annual report, Calder Industries reported full-year "
            "revenue of 412 million dollars.",
            metadata={"source": "2019-annual-report.txt"})
        before_restatement = datetime.now(timezone.utc).isoformat()
        print(f"watermark taken: {before_restatement}")

        await rag.index_document(
            "Calder Industries restated its 2019 revenue to 388 million "
            "dollars following an accounting review.",
            metadata={"source": "2021-restatement.txt"})
        print("restatement indexed")

        banner("What is 2019 revenue? (today's belief)")
        print((await rag.query("What was Calder Industries' 2019 revenue?"))["answer"])

        banner("What did we believe before the restatement was filed?")
        out = await rag.query(
            "What was Calder Industries' 2019 revenue?",
            param=QueryParam(as_believed_at=before_restatement))
        print(out["answer"])
        print("\n-> The pre-restatement answer is reproducible on demand.")
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
