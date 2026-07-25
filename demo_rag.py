"""Demo script demonstrating post-graph-rag pipeline end-to-end with DocumentMetadata."""
import asyncio
import os
from post_graph_rag import GraphRAG, RAGConfig, DocumentMetadata

DB_URI = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")
MODEL_ROUTER_URL = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")

async def main():
    print("=" * 60)
    print("POST-GRAPH-RAG DOCUMENT METADATA DEMO")
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

    print("\n[+] Indexing Document 1 with DocumentMetadata...")
    res1 = await rag.index_document(sample_doc_1, metadata=meta1)
    print(f"    Indexed Doc 1: Extracted {res1['entities_extracted']} entities, {res1['triples_extracted']} triples.")
    print(f"    Metadata Payload: {res1['metadata']}")

    print("\n[+] Indexing Document 2 with DocumentMetadata...")
    res2 = await rag.index_document(sample_doc_2, metadata=meta2)
    print(f"    Indexed Doc 2: Extracted {res2['entities_extracted']} entities, {res2['triples_extracted']} triples.")
    print(f"    Metadata Payload: {res2['metadata']}")

    print("\n[+] Querying GraphRAG Engine...")
    query_text = "Who are the parents of Zeus and Poseidon, and what domains do they rule?"
    answer_res = await rag.query(query_text, top_k=3)

    print("\n" + "-" * 50)
    print(f"QUESTION: {answer_res['question']}")
    print("-" * 50)
    print(f"ANSWER:\n{answer_res['answer']}\n")
    print("RETRIEVED DOCUMENTS & METADATA:")
    for doc in answer_res['retrieved_documents']:
        print(f"  - ID={doc['id']} | Metadata={doc['metadata']}")
        print(f"    Text: {doc['text'][:80]}...")
    print("\nRETRIEVED ENTITIES:")
    for entity in answer_res['retrieved_entities']:
        print(f"  - {entity}")
    print("\nRETRIEVED GRAPH TRIPLES:")
    for triple in answer_res['retrieved_graph_triples']:
        print(f"  - {triple}")
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
