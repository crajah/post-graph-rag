"""A passage with nothing extractable is still a passage.

Refusing to write placeholder *structure* is correct and stays: inventing
entities corrupts the graph, and a weak extraction model does not fail loudly,
it degrades quietly. But the chunk's embedding never needed entities. A page of
dialogue naming nobody is still a page a reader should reach by meaning, and
dropping it leaves text that is in the corpus and retrievable by nothing --
indistinguishable from text that was never ingested.

The distinction these tests defend: an entity-free passage and a dead
extraction model are not the same event, and must not become the same event.
"""
import pytest
from conftest import fake_embed

from post_graph_rag import DocumentMetadata
from post_graph_rag.errors import ExtractionError, LLMError
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

# Returned by the fake LLM as a well-formed but empty result, which is what the
# extractor turns into ExtractionError -- the live symptom being fixed.
EMPTY = ExtractionResult(entities=[], triples=[])

RICH = ExtractionResult(
    entities=[Entity(name="Zeus", type="Person", description="king of the gods")],
    triples=[Triple(subject="Zeus", predicate="rules", object="Olympus",
                    description="dominion")],
)

DIALOGUE = "and then she said it was nothing at all, and he said nothing back."


async def _entities_in(rag, space="default"):
    ents = await rag.store.client.get_vertices("entities", realm=rag.config.realm, space=space)
    return ents


async def _docs_in(rag, space="default"):
    return await rag.store.client.get_vertices("documents", realm=rag.config.realm, space=space)


@pytest.mark.asyncio
async def test_an_entity_free_chunk_is_stored_and_retrievable(rag_factory):
    """The reported bug: the passage went with the structure."""
    rag = await rag_factory(extraction=EMPTY)

    results = await rag.index_documents([(DIALOGUE, DocumentMetadata(document="play.txt"))])

    assert len(results) == 1, "the chunk must be indexed, not skipped"
    assert results[0]["entities_extracted"] == 0
    assert results[0]["triples_extracted"] == 0
    assert results[0]["relations_added"] == 0

    docs = await _docs_in(rag)
    assert len(docs) == 1, "the passage must be in the corpus"

    hits = await rag.store.search_similar_documents(fake_embed(DIALOGUE), top_k=5, space="default")
    assert hits, "the passage must be retrievable by meaning"
    assert any(DIALOGUE in (v.payload or {}).get("text", "") for v, _ in hits)


@pytest.mark.asyncio
async def test_the_result_says_the_extraction_was_empty(rag_factory):
    """A caller has to be able to count these and tell 'no entities found'
    apart from 'not indexed'."""
    rag = await rag_factory(extraction=EMPTY)
    results = await rag.index_documents([(DIALOGUE, DocumentMetadata(document="play.txt"))])
    assert results[0]["extraction_empty"] is True

    rag2 = await rag_factory(extraction=RICH)
    ok = await rag2.index_documents([("Zeus rules Olympus.", DocumentMetadata(document="myth.txt"))])
    assert ok[0]["extraction_empty"] is False


@pytest.mark.asyncio
async def test_any_other_failure_still_skips_the_chunk(rag_factory):
    """A timeout, a connection error and an entity-free passage are not the
    same thing. Collapsing them would let a dead extraction model read as a
    corpus with no entities in it, silently."""
    rag = await rag_factory(extraction=RICH)

    async def boom(*a, **k):
        raise LLMError("extraction endpoint is down")

    rag.extractor.extract_from_text = boom

    with pytest.raises(LLMError):
        await rag.index_documents([("Zeus rules Olympus.", DocumentMetadata(document="myth.txt"))])
    assert await _docs_in(rag) == [], "a failed extraction must not store the chunk"


@pytest.mark.asyncio
async def test_one_bad_chunk_does_not_take_the_entity_free_ones_with_it(rag_factory):
    """Isolation still holds: the transient failure is skipped, the entity-free
    passages are kept."""
    rag = await rag_factory(extraction=EMPTY)
    calls = {"n": 0}
    real = rag.extractor.extract_from_text

    async def flaky(text, context=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise LLMError("transient")
        return await real(text, context=context)

    rag.extractor.extract_from_text = flaky
    results = await rag.index_documents([
        (f"{DIALOGUE} one", DocumentMetadata(document="p.txt")),
        (f"{DIALOGUE} two", DocumentMetadata(document="p.txt")),
        (f"{DIALOGUE} three", DocumentMetadata(document="p.txt")),
    ])
    assert len(results) == 2
    assert len(await _docs_in(rag)) == 2


@pytest.mark.asyncio
async def test_a_document_with_no_entities_anywhere_returns_results(rag_factory):
    """_assert_any_progress must not treat 'every chunk was entity-free' as an
    outage: every chunk succeeded."""
    rag = await rag_factory(extraction=EMPTY)
    results = await rag.index_text(
        "\n\n".join(f"{DIALOGUE} {i}" for i in range(3)),
        metadata=DocumentMetadata(document="play.txt"),
    )
    assert len(results) >= 1
    assert all(r["entities_extracted"] == 0 for r in results)
    assert len(await _docs_in(rag)) == len(results)


@pytest.mark.asyncio
async def test_nothing_indexed_at_all_still_raises(rag_factory):
    """The outage guard survives the fix."""
    rag = await rag_factory(extraction=RICH)

    async def boom(*a, **k):
        raise LLMError("endpoint down")

    rag.extractor.extract_from_text = boom
    with pytest.raises(LLMError):
        await rag.index_documents([
            ("Zeus rules Olympus.", DocumentMetadata(document="a.txt")),
            ("Hera rules Argos.", DocumentMetadata(document="b.txt")),
        ])


@pytest.mark.asyncio
async def test_the_graph_stays_clean(rag_factory):
    """No placeholder entities, no empty triples, no dangling relation."""
    rag = await rag_factory(extraction=EMPTY)
    await rag.index_documents([(DIALOGUE, DocumentMetadata(document="play.txt"))])

    assert await _entities_in(rag) == [], "no placeholder structure may be written"

    rels = await rag.store.get_all_relations(space="default")
    assert rels == []

    ref = rag.store.client._get_table_ref("relations", rag.config.realm)
    rows = await rag.store.client._fetch(
        f"SELECT count(*) AS n FROM {ref} WHERE realm = $1", rag.config.realm)
    assert int(rows[0]["n"]) == 0


@pytest.mark.asyncio
async def test_index_document_stores_the_passage_too(rag_factory):
    """The single-chunk public method behaves the same way, deliberately: a
    caller looping over chunks with index_document hits the identical bug, and
    an exception discards an embedding that was already paid for."""
    rag = await rag_factory(extraction=EMPTY)
    res = await rag.index_document(DIALOGUE, metadata=DocumentMetadata(document="play.txt"))
    assert res["entities_extracted"] == 0
    assert res["extraction_empty"] is True
    assert len(await _docs_in(rag)) == 1


@pytest.mark.asyncio
async def test_index_document_still_raises_on_a_real_failure(rag_factory):
    rag = await rag_factory(extraction=RICH)

    async def boom(*a, **k):
        raise LLMError("endpoint down")

    rag.extractor.extract_from_text = boom
    with pytest.raises(LLMError):
        await rag.index_document("Zeus rules Olympus.", metadata=DocumentMetadata(document="a.txt"))
    assert await _docs_in(rag) == []


@pytest.mark.asyncio
async def test_extraction_error_is_still_raised_by_the_extractor(rag_factory):
    """The refusal itself is unchanged; only the engine's response to it moves.
    Callers using the extractor directly keep the exception."""
    rag = await rag_factory(extraction=EMPTY)
    with pytest.raises(ExtractionError):
        await rag.extractor.extract_from_text(DIALOGUE)
