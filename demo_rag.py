"""Demo: indexing, entity resolution across documents, and structured retrieval.

Requires PostgreSQL with pgvector and an OpenAI-compatible endpoint:

    export POSTGRES_URI="postgresql://user@localhost:5432/postgres"
    export OPENAI_API_BASE="http://localhost:4000/v1"
    export OPENAI_API_KEY="..."
    python demo_rag.py
"""
import asyncio
import os

from post_graph_rag import DocumentMetadata, GraphRAG, QueryParam, RAGConfig, RAGError

REALM = os.getenv("RAG_DEMO_REALM", "demo_realm")


async def main():
    print("=" * 62)
    print("POST-GRAPH-RAG DEMO")
    print("=" * 62)

    config = RAGConfig(
        model=os.getenv("RAG_MODEL", "DeepSeek-V3.2"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.getenv("RAG_EMBEDDING_DIM", "1536")),
        realm=REALM,
        # Each realm gets its own schema, so the demo cannot collide with
        # existing tables in the target database.
        schema_per_realm=True,
    )

    rag = GraphRAG(config)
    print(f"\n[+] Connecting to {config.db_uri} (realm={config.realm})")
    await rag.initialize()

    zeus_doc = (
        "Zeus is the king of the Olympian gods, ruling sky and thunder from Mount Olympus. "
        "He is the son of Cronus and Rhea, and married to Hera. "
        "Zeus defeated the Titans in the Titanomachy to establish his rule."
    )
    poseidon_doc = (
        "Poseidon is the Greek god of the sea, earthquakes, and horses. "
        "He is the brother of Zeus and Hades, born to Cronus and Rhea. "
        "Poseidon wields a mighty trident and built underwater palaces in the Aegean Sea."
    )

    try:
        print("\n[+] Indexing document 1...")
        res1 = await rag.index_document(zeus_doc, metadata=DocumentMetadata(
            source="https://mythology.org/zeus.html",
            category="greek_mythology",
            collection="olympian_deities",
            document="zeus_overview.pdf",
            page=1,
            paragraph=1,
        ))
        print(f"    {res1['entities_extracted']} entities, {res1['triples_extracted']} triples, "
              f"{res1['mentions_added']} mentions")

        print("\n[+] Indexing document 2...")
        res2 = await rag.index_document(poseidon_doc, metadata=DocumentMetadata(
            source="https://mythology.org/poseidon.html",
            category="greek_mythology",
            collection="olympian_deities",
            document="poseidon_overview.pdf",
            page=3,
            paragraph=2,
        ))
        print(f"    {res2['entities_extracted']} entities, {res2['triples_extracted']} triples, "
              f"{res2['mentions_added']} mentions")

        # Cronus and Rhea appear in both documents. Entity resolution means they
        # are one vertex each, which is what lets the graph connect the two chunks.
        rows = await rag.store.client._fetch(
            f'SELECT payload->>\'name\' AS name, count(*) AS n FROM "{REALM}"."entities" '
            "GROUP BY 1 HAVING count(*) > 1"
        )
        print(f"\n[+] Duplicate entity vertices: {len(rows)} (expected 0)")

        shared = await rag.store.find_entity_by_name("Cronus")
        if shared:
            inbound = await rag.store.client._fetch(
                f'SELECT count(*) AS n FROM "{REALM}"."relations" WHERE to_id = $1',
                int(shared.id),
            )
            print(f"    'Cronus' is a single vertex (id={shared.id}) reached by "
                  f"{inbound[0]['n']} relations from both documents")

        query = "Who are the parents of Zeus and Poseidon, and what do they rule?"

        print("\n[+] Structured retrieval via query_data()...")
        data = await rag.query_data(query, param=QueryParam(mode="mix", top_k=3))
        print(f"    keywords:      {data['metadata']['keywords']}")
        print(f"    entities:      {[e['entity_name'] for e in data['data']['entities']]}")
        print(f"    chunks:        {[c['metadata'].get('document') for c in data['data']['chunks']]}")
        print("    relationships:")
        for r in data["data"]["relationships"][:8]:
            print(f"      ({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})")

        print("\n[+] Full synthesis via query()...")
        res = await rag.query(query, param=QueryParam(mode="mix", top_k=3))
        print("-" * 62)
        print(res["answer"])
        print("-" * 62)
        print("References:", res["references"])

    except RAGError as e:
        # Failures are raised rather than degrading into unranked results.
        print(f"\n[!] {type(e).__name__}: {e}")
    finally:
        print("\n[+] Dropping demo schema...")
        await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{REALM}" CASCADE;')
        await rag.close()
        print("[+] Done.")


if __name__ == "__main__":
    asyncio.run(main())
