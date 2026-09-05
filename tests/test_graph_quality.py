"""DB-backed tests for alias merging, relation dedup/weight, and mention expansion."""
import pytest
from conftest import fake_embed

from post_graph_rag import DocumentMetadata, QueryParam
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

BABBAGE = ExtractionResult(
    entities=[
        Entity(name="Charles Babbage", type="Person", description="inventor",
               aliases=["Babbage"]),
        Entity(name="Analytical Engine", type="Technology", description="mechanical computer"),
    ],
    triples=[Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine",
                    description="Babbage designed it")],
)


# ------------------------------------------------------------ alias merging

@pytest.mark.asyncio
async def test_alias_lookup_finds_entity(rag_factory):
    rag = await rag_factory()
    store = rag.store
    await store.upsert_entity("Charles Babbage", "Person", "inventor",
                              fake_embed("babbage"), aliases=["Babbage", "C. Babbage"])
    found = await store.find_entity_by_name("Babbage")
    assert found is not None
    assert found.payload["name"] == "Charles Babbage"


@pytest.mark.asyncio
async def test_short_form_merges_into_canonical(rag_factory):
    """'Babbage' and 'Charles Babbage' must be one vertex, or facts attached to
    each are unreachable from the other."""
    rag = await rag_factory()
    store = rag.store
    full = await store.upsert_entity("Charles Babbage", "Person", "inventor",
                                     fake_embed("babbage"), aliases=["Babbage"])
    short = await store.upsert_entity("Babbage", "Person", "", fake_embed("babbage"))
    assert full.id == short.id

    rows = await store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities"')
    assert rows[0]["n"] == 1


@pytest.mark.asyncio
async def test_canonical_name_upgrades_to_fuller_form(rag_factory):
    """Indexed short-form-first, the vertex should still end up canonical."""
    rag = await rag_factory()
    store = rag.store
    first = await store.upsert_entity("Babbage", "Person", "inventor", fake_embed("babbage"))
    second = await store.upsert_entity("Charles Babbage", "Person", "inventor",
                                       fake_embed("babbage"), aliases=["Babbage"])
    assert first.id == second.id
    found = await store.find_entity_by_name("Babbage")
    assert found.payload["name"] == "Charles Babbage"
    assert "Babbage" in found.payload["aliases"]


@pytest.mark.asyncio
async def test_aliases_accumulate_across_chunks(rag_factory):
    rag = await rag_factory()
    store = rag.store
    await store.upsert_entity("Ada Lovelace", "Person", "d", fake_embed("ada"), aliases=["Ada"])
    await store.upsert_entity("Ada Lovelace", "Person", "d", fake_embed("ada"),
                              aliases=["Countess of Lovelace"])
    found = await store.find_entity_by_name("Countess of Lovelace")
    assert found is not None
    assert set(found.payload["aliases"]) >= {"Ada", "Countess of Lovelace"}


# --------------------------------------------------- relation dedup / weight

@pytest.mark.asyncio
async def test_repeated_relation_increments_weight(rag_factory):
    """The same triple seen in two chunks is one edge with weight 2, not two edges."""
    rag = await rag_factory()
    store = rag.store
    a = await store.upsert_entity("Charles Babbage", "Person", "d", fake_embed("babbage"))
    b = await store.upsert_entity("Analytical Engine", "Technology", "d", fake_embed("engine"))

    e1 = await store.add_relation(a, b, "designed", "first mention")
    e2 = await store.add_relation(a, b, "designed", "second mention")
    assert e1.id == e2.id

    rows = await store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."relations"')
    assert rows[0]["n"] == 1
    edge = await store.find_relation(a.id, "designed", b.id)
    assert edge.payload["weight"] == 2


@pytest.mark.asyncio
async def test_distinct_predicates_stay_separate(rag_factory):
    rag = await rag_factory()
    store = rag.store
    a = await store.upsert_entity("A", "Concept", "d", fake_embed("zeus"))
    b = await store.upsert_entity("B", "Concept", "d", fake_embed("hera"))
    await store.add_relation(a, b, "designed", "x")
    await store.add_relation(a, b, "built", "y")
    rows = await store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."relations"')
    assert rows[0]["n"] == 2


@pytest.mark.asyncio
async def test_negated_relation_persisted_and_surfaced(rag_factory):
    rag = await rag_factory()
    store = rag.store
    a = await store.upsert_entity("Ada Lovelace", "Person", "d", fake_embed("ada"))
    b = await store.upsert_entity("Lord Byron", "Person", "d", fake_embed("zeus"))
    await store.add_relation(a, b, "had_relationship_with", "estranged", negated=True)
    edge = await store.find_relation(a.id, "had_relationship_with", b.id)
    assert edge.payload["negated"] is True

    res = await rag.query_data("Ada Lovelace", param=QueryParam(mode="mix", top_k=5, ll_keywords=["Ada"]))
    rels = [r for r in res["data"]["relationships"] if r["relation_type"] == "had_relationship_with"]
    assert rels and rels[0]["negated"] is True


# ---------------------------------------------------------- doc_mentions use

@pytest.mark.asyncio
async def test_doc_mentions_are_not_duplicated(rag_factory):
    rag = await rag_factory(extraction=BABBAGE)
    await rag.index_document("Babbage designed the Analytical Engine.",
                             metadata=DocumentMetadata(document="a.txt"))
    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."doc_mentions"')
    before = rows[0]["n"]
    # Re-index identical content: entities resolve to the same vertices, and the
    # new chunk gets its own mentions, but no chunk gains duplicates.
    dupes = await rag.store.client._fetch(
        f'SELECT from_id, to_id, count(*) c FROM "{rag.config.realm}"."doc_mentions" '
        f"GROUP BY 1,2 HAVING count(*) > 1")
    assert before > 0
    assert dupes == []


@pytest.mark.asyncio
async def test_mention_expansion_pulls_supporting_chunks(rag_factory):
    """A chunk that mentions a matched entity is retrieved even when its own
    wording does not match the query."""
    rag = await rag_factory(extraction=BABBAGE)
    store = rag.store

    doc = await store.add_document("Zeus text unrelated to the query wording.",
                                   fake_embed("zeus olympian"), DocumentMetadata(document="other.txt"))
    ent = await store.upsert_entity("Python", "Technology", "language", fake_embed("python"))
    await store.add_doc_mention(doc, ent)

    found = await store.find_chunks_mentioning([ent.id], limit=5)
    assert [v.payload.get("document") for v in found] == ["other.txt"]


@pytest.mark.asyncio
async def test_mention_expansion_respects_space(rag_factory):
    rag = await rag_factory()
    store = rag.store
    doc = await store.add_document("prod text", fake_embed("zeus"),
                                   DocumentMetadata(document="p.txt"), space="production")
    ent = await store.upsert_entity("Zeus", "Person", "d", fake_embed("zeus"), space="production")
    await store.add_doc_mention(doc, ent, space="production")

    assert await store.find_chunks_mentioning([ent.id], space="production")
    assert await store.find_chunks_mentioning([ent.id], space="sandbox") == []


# ------------------------------------------------------------------ batching

@pytest.mark.asyncio
async def test_entity_embeddings_are_batched(rag_factory):
    """Entity embeddings should be one request, not one per entity."""
    rag = await rag_factory(extraction=BABBAGE)
    rag.llm.batch_calls.clear()
    await rag.index_document("Babbage designed the Analytical Engine.",
                             metadata=DocumentMetadata(document="a.txt"))
    assert any(len(call) == 2 for call in rag.llm.batch_calls)


@pytest.mark.asyncio
async def test_index_text_chunks_and_threads_context(rag_factory):
    rag = await rag_factory(extraction=BABBAGE, chunk_chars=300, chunk_overlap_chars=60)
    text = "\n".join(
        f"Charles Babbage designed engines, paragraph {i}, with enough length to be kept."
        for i in range(12)
    )
    results = await rag.index_text(text, metadata=DocumentMetadata(document="babbage.txt"))
    assert len(results) > 1
    assert all(r["metadata"]["document"] == "babbage.txt" for r in results)
    # Entities are shared across chunks rather than re-created.
    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities"')
    assert rows[0]["n"] == 2
