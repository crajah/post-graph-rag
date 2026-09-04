"""The same question through each graph retrieval path, side by side.

  local   entity-centred: find the entities named, walk their relations
  global  community reports: corpus-level themes rather than specific facts
  mix     all channels fused with reciprocal rank fusion (the default)

The modes are not ranked -- they answer differently. A question naming
specific entities is served by `local`; "what are the themes here" wants
`global`; `mix` is the general default. Run this on your own corpus to see
which shape your questions take.
"""
import asyncio

from _shared import banner, fresh_realm, make_config
from post_graph_rag import GraphRAG, QueryParam

DOCS = [
    ("2020.txt", "Lumen Materials was headquartered in Bristol and led by "
                 "CEO Dana Whitfield. It employed 300 people."),
    ("2022.txt", "Lumen Materials relocated its headquarters to Cardiff. "
                 "Dana Whitfield remained chief executive."),
    ("2024.txt", "Lumen Materials appointed Tobias Reyes as chief executive, "
                 "succeeding Dana Whitfield. Headcount reached 1,100."),
]


async def main():
    rag = GraphRAG(make_config(fresh_realm("example_modes")))
    await rag.initialize()
    try:
        for source, text in DOCS:
            await rag.index_document(text, metadata={"source": source})
        await rag.build_communities()

        question = ("Where is Lumen Materials based, who runs it, and how has "
                    "that changed?")
        for mode in ("local", "global", "mix"):
            banner(f"mode = {mode}")
            out = await rag.query(question, param=QueryParam(mode=mode, top_k=8))
            print(out["answer"])
    finally:
        await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
