"""Demo script demonstrating post-graph-rag query_data API and dual-level keyword extraction."""
import asyncio
import os
from post_graph_rag import GraphRAG, RAGConfig, DocumentMetadata, QueryParam

DB_URI = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")
MODEL_ROUTER_URL = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")

async def main():
    print("=" * 60)
    print("POST-GRAPH-RAG LIGHTRAG-INSPIRED PIPELINE DEMO")
    print("=" * 60)

    config = RAGConfig(
        api_base=MODEL_ROUTER_URL,
        api_key=os.getenv("OPENAI_API_KEY", "BEVZ-6L81-OZ8Y"),
        model="DeepSeek-V3.2",
        embedding_model="text-embedding-3-small",
        embedding_dim=4,
        db_uri=DB_URI,
        realm="demo_realm"
    )

    rag = GraphRAG(config)
    print(f"\n[+] Connecting to Postgres: {DB_URI}")
    await rag.initialize()

    sample_doc_1 = (
        "Zeus is the king of the Olympian gods, ruling sky and thunder from Mount Olympus. "
        "He is the son of Cronus and Rhea, and married to Hera. "
        "Zeus defeated the Titans in the Titanomachy to establish his rule over mortals and gods."
    )

    sample_doc_2 = (
        "Poseidon is the Greek god of the sea, earthquakes, and horses. "
        "He is the brother of Zeus and Hades, born to Cronus and Rhea. "
        "Poseidon wields a mighty trident and built underwater palaces in the Aegean Sea."
    )

    meta1 = DocumentMetadata(
        source="https://mythology.org/zeus.html",
        category="greek_mythology",
        collection="olympian_deities",
        document="zeus_overview.pdf",
        page=1,
        paragraph=1
    )

    meta2 = DocumentMetadata(
        source="https://mythology.org/poseidon.html",
        category="greek_mythology",
        collection="olympian_deities",
        document="poseidon_overview.pdf",
        page=3,
        paragraph=2
    )

    print("\n[+] Indexing Document 1...")
    res1 = await rag.index_document(sample_doc_1, metadata=meta1)
    print(f"    Indexed Doc 1: Extracted {res1['entities_extracted']} entities, {res1['triples_extracted']} triples.")

    print("\n[+] Indexing Document 2...")
    res2 = await rag.index_document(sample_doc_2, metadata=meta2)
    print(f"    Indexed Doc 2: Extracted {res2['entities_extracted']} entities, {res2['triples_extracted']} triples.")

    query_text = "Who are the parents of Zeus and Poseidon, and what domains do they rule?"

    print("\n[+] Testing Dual-Level Keyword Extraction...")
    kw_res = await rag.extractor.extract_keywords(query_text)
    print(f"    High-Level Keywords: {kw_res.high_level_keywords}")
    print(f"    Low-Level Keywords: {kw_res.low_level_keywords}")

    print("\n[+] Testing query_data() Structured Retrieval API...")
    data_res = await rag.query_data(query_text, param=QueryParam(mode="mix", top_k=2))
    print(f"    Status: {data_res['status']}")
    print(f"    Query Mode: {data_res['metadata']['query_mode']}")
    print(f"    Entities Found: {len(data_res['data']['entities'])}")
    print(f"    Relationships Found: {len(data_res['data']['relationships'])}")
    print(f"    Chunks Found: {len(data_res['data']['chunks'])}")
    print(f"    References Generated: {data_res['data']['references']}")

    print("\n[+] Querying Full GraphRAG Engine with Citations...")
    param = QueryParam(mode="mix", top_k=2)
    res = await rag.query(query_text, param=param)
    print("\n" + "-" * 50)
    print(f"QUESTION: {res['question']}")
    print("-" * 50)
    print(f"ANSWER:\n{res['answer']}\n")
    print("REFERENCES:", res['references'])
    print("-" * 50)

    print("\n[+] Cleaning up demo graph tables...")
    for tbl in ["relations", "doc_mentions", "entities", "documents"]:
        t_ref = rag.store.client._get_table_ref(tbl, realm=config.realm)
        a_ref = rag.store.client._get_table_ref(f"{tbl}_audit", realm=config.realm)
        d_ref = rag.store.client._get_table_ref(f"{tbl}_data", realm=config.realm)
        await rag.store.client._execute(f"DROP TABLE IF EXISTS {d_ref} CASCADE;")
        await rag.store.client._execute(f"DROP TABLE IF EXISTS {a_ref} CASCADE;")
        await rag.store.client._execute(f"DROP TABLE IF EXISTS {t_ref} CASCADE;")

    await rag.close()
    print("[+] Cleanup complete. Demo finished!")

if __name__ == "__main__":
    asyncio.run(main())
