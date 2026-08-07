"""End-to-end tests for the GraphRAG engine against a real database."""
import pytest

from post_graph_rag import DocumentMetadata, QueryParam
from post_graph_rag.errors import ExtractionError
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

ZEUS_DOC = (
    "Zeus is king of the Olympian gods, son of Cronus and Rhea, married to Hera."
)
PYTHON_DOC = (
    "The Python programming language was created by Guido van Rossum."
)

ZEUS_EXTRACTION = ExtractionResult(
    entities=[
        Entity(name="Zeus", type="Person", description="king of the Olympian gods"),
        Entity(name="Hera", type="Person", description="wife of Zeus"),
        Entity(name="Cronus", type="Person", description="father of Zeus"),
    ],
    triples=[
        Triple(subject="Zeus", predicate="married_to", object="Hera", description="spouse"),
        Triple(subject="Zeus", predicate="son_of", object="Cronus", description="parentage"),
    ],
)

PYTHON_EXTRACTION = ExtractionResult(
    entities=[
        Entity(name="Python", type="Software", description="programming language"),
        Entity(name="Guido van Rossum", type="Person", description="created Python"),
    ],
    triples=[
        Triple(subject="Python", predicate="created_by", object="Guido van Rossum", description="authorship"),
    ],
)


@pytest.mark.asyncio
async def test_index_document_reports_real_counts(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    res = await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))
    assert res["entities_extracted"] == 3
    assert res["triples_extracted"] == 2
    assert res["relations_added"] == 2
    assert res["mentions_added"] == 3


@pytest.mark.asyncio
async def test_doc_mentions_edges_are_written(rag_factory):
    """The doc_mentions table was created by the schema but never written to."""
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))
    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."doc_mentions"'
    )
    assert rows[0]["n"] == 3


@pytest.mark.asyncio
async def test_entities_shared_across_documents(rag_factory):
    """Cross-document entity connection is the whole point of the graph: Zeus
    mentioned in two chunks must be one vertex, not two."""
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="a.pdf"))
    await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="b.pdf"))

    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities" '
        "WHERE lower(payload->>'name') = 'zeus'"
    )
    assert rows[0]["n"] == 1


@pytest.mark.asyncio
async def test_retrieval_is_relevance_ranked(rag_factory):
    """The headline regression.

    Asking about Python used to return the Zeus chunk first, because vector
    search silently failed and the fallback returned rows in insertion order.
    """
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))

    rag.llm._extraction = PYTHON_EXTRACTION
    await rag.index_document(PYTHON_DOC, metadata=DocumentMetadata(document="python.pdf"))

    res = await rag.query_data(
        "Who created the Python programming language?",
        param=QueryParam(mode="mix", top_k=1, hl_keywords=["programming"], ll_keywords=["Python"]),
    )
    docs = [c["metadata"].get("document") for c in res["data"]["chunks"]]
    assert docs == ["python.pdf"]


@pytest.mark.asyncio
async def test_extraction_failure_aborts_indexing(rag_factory):
    """No partial graph of fabricated edges on LLM failure."""
    rag = await rag_factory(extraction=ExtractionResult())
    with pytest.raises(ExtractionError):
        await rag.index_document(ZEUS_DOC)

    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities"'
    )
    assert rows[0]["n"] == 0


@pytest.mark.asyncio
async def test_relations_are_specific_not_generic(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC)
    rows = await rag.store.client._fetch(
        f'SELECT DISTINCT relation_type FROM "{rag.config.realm}"."relations"'
    )
    types = sorted(r["relation_type"] for r in rows)
    assert types == ["married_to", "son_of"]
    assert "relates_to" not in types


@pytest.mark.asyncio
async def test_query_data_includes_graph_triples(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC)
    res = await rag.query_data(
        "Who is Zeus married to?",
        param=QueryParam(mode="mix", top_k=5, ll_keywords=["Zeus"]),
    )
    rels = {(r["src_id"], r["relation_type"], r["tgt_id"]) for r in res["data"]["relationships"]}
    assert ("Zeus", "married_to", "Hera") in rels


@pytest.mark.asyncio
async def test_unknown_mode_rejected(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    with pytest.raises(ValueError):
        await rag.query_data("anything", param=QueryParam(mode="nonsense"))


@pytest.mark.asyncio
async def test_bypass_mode_skips_retrieval(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION, answer="direct answer")
    res = await rag.query("hello", param=QueryParam(mode="bypass"))
    assert res["answer"] == "direct answer"
    assert res["retrieved_documents"] == []


@pytest.mark.asyncio
async def test_streaming_query_yields_tokens(rag_factory):
    """Regression: QueryParam(stream=True) raised AttributeError because
    chat_completion_stream did not exist."""
    rag = await rag_factory(extraction=ZEUS_EXTRACTION, answer="streamed answer here")
    await rag.index_document(ZEUS_DOC)
    stream = await rag.query("Who is Zeus?", param=QueryParam(mode="mix", stream=True))
    chunks = [c async for c in stream]
    assert "".join(chunks).strip() == "streamed answer here"


@pytest.mark.asyncio
async def test_keywords_influence_entity_retrieval(rag_factory):
    """Keywords used to be extracted and then discarded into metadata."""
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC)
    rag.llm.embed_calls.clear()

    await rag.query_data(
        "Tell me about the family",
        param=QueryParam(mode="mix", top_k=2, hl_keywords=["mythology"], ll_keywords=["Hera"]),
    )
    assert any("Hera" in call for call in rag.llm.embed_calls)


@pytest.mark.asyncio
async def test_spaces_isolate_query_results(rag_factory):
    rag = await rag_factory(extraction=ZEUS_EXTRACTION)
    await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="prod.pdf"), space="production")

    rag.llm._extraction = PYTHON_EXTRACTION
    await rag.index_document(PYTHON_DOC, metadata=DocumentMetadata(document="sand.pdf"), space="sandbox")

    res = await rag.query_data(
        "Zeus", param=QueryParam(mode="mix", top_k=5, space="sandbox", ll_keywords=["Zeus"])
    )
    docs = [c["metadata"].get("document") for c in res["data"]["chunks"]]
    assert docs == ["sand.pdf"]
