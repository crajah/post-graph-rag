"""One realm, many tenants: spaces keep corpora apart and let you cross them.

Each customer's documents go in their own space. A query scoped to a space
sees only that tenant. The same graph can then answer a cross-tenant question
when you deliberately ask for one -- useful for portfolio or sector views,
and the reason spaces are a query parameter rather than a separate database.
"""
import asyncio

from _shared import banner, fresh_realm, make_config
from post_graph_rag import GraphRAG, QueryParam
from post_graph import RESERVED_SPACE_ALL

TENANTS = {
    "acme": "Acme Corp reported quarterly revenue of 24 million dollars and "
            "operates two manufacturing plants in Ohio.",
    "globex": "Globex Inc reported quarterly revenue of 31 million dollars "
              "and operates a single plant in Arizona.",
}


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_spaces")))
    await rag.initialize()
    try:
        for space, text in TENANTS.items():
            await rag.index_document(text, metadata={"source": f"{space}-q3.txt"},
                                     space=space)
            print(f"indexed into space '{space}'")

        banner("Scoped to acme -- globex is invisible")
        out = await rag.query("What was quarterly revenue?",
                              param=QueryParam(space="acme"))
        print(out["answer"])

        banner("Across every space -- a deliberate cross-tenant view")
        out = await rag.query(
            "Compare quarterly revenue across all companies.",
            param=QueryParam(space=RESERVED_SPACE_ALL, top_k=20))
        print(out["answer"])
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
