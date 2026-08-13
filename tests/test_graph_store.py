"""DB-backed tests for RAGGraphStore: schema strictness, entity resolution, spaces."""
import pytest

from post_graph_rag import RAGGraphStore
from post_graph_rag.errors import SchemaError

from conftest import VOCAB_DIM, fake_embed, make_config


@pytest.mark.asyncio
async def test_schema_creates_embedding_columns(rag_factory):
    rag = await rag_factory()
    client = rag.store.client
    for table in ("documents", "entities", "documents_data", "entities_data"):
        rows = await client._fetch(
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped",
            f'"{rag.config.realm}"."{table}"',
        )
        assert rows and rows[0]["t"] == f"vector({VOCAB_DIM})", table


@pytest.mark.asyncio
async def test_initialize_schema_is_idempotent(rag_factory):
    rag = await rag_factory()
    await rag.store.initialize_schema()


@pytest.mark.asyncio
async def test_embedding_dim_mismatch_raises(rag_factory):
    """A pre-existing table with a different vector width used to be accepted
    silently, after which every search returned nothing."""
    rag = await rag_factory()
    clashing = RAGGraphStore(make_config(realm=rag.config.realm, embedding_dim=VOCAB_DIM + 8))
    await clashing.connect()
    try:
        with pytest.raises(SchemaError):
            await clashing.initialize_schema()
    finally:
        await clashing.close()


@pytest.mark.asyncio
async def test_upsert_entity_resolves_by_name(rag_factory):
    """Without this every mention creates a new vertex, so the same entity in
    two documents becomes two disconnected nodes."""
    rag = await rag_factory()
    store = rag.store
    emb = fake_embed("zeus")
    a = await store.upsert_entity("Zeus", "Person", "king of the gods", emb)
    b = await store.upsert_entity("Zeus", "Person", "king of the gods", emb)
    c = await store.upsert_entity("ZEUS", "Person", "king of the gods", emb)
    assert a.id == b.id == c.id

    rows = await store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities" '
        "WHERE lower(payload->>'name') = 'zeus'"
    )
    assert rows[0]["n"] == 1


@pytest.mark.asyncio
async def test_placeholder_does_not_clobber_real_entity(rag_factory):
    """Triple endpoints are upserted as bare 'Concept' stubs; they must not
    overwrite a richer type/description recorded earlier."""
    rag = await rag_factory()
    store = rag.store
    emb = fake_embed("zeus")
    await store.upsert_entity("Zeus", "Person", "king of the gods", emb)
    await store.upsert_entity("Zeus", "Concept", "", emb)

    found = await store.find_entity_by_name("Zeus")
    assert found.payload["type"] == "Person"
    assert found.payload["description"] == "king of the gods"


@pytest.mark.asyncio
async def test_entities_are_isolated_per_space(rag_factory):
    rag = await rag_factory()
    store = rag.store
    emb = fake_embed("zeus")
    prod = await store.upsert_entity("Zeus", "Person", "prod", emb, space="production")
    sand = await store.upsert_entity("Zeus", "Person", "sandbox", emb, space="sandbox")
    assert prod.id != sand.id
    assert (await store.find_entity_by_name("Zeus", space="production")).id == prod.id


@pytest.mark.asyncio
async def test_get_neighbors_is_space_scoped(rag_factory):
    """Vector search is space-scoped, so traversal from a matched entity must be
    too, or results leak across tenants."""
    rag = await rag_factory()
    store = rag.store
    e = fake_embed("zeus")
    h = fake_embed("hera")

    z_prod = await store.upsert_entity("Zeus", "Person", "d", e, space="production")
    h_prod = await store.upsert_entity("Hera", "Person", "d", h, space="production")
    await store.add_relation(z_prod, h_prod, "married_to", "spouse", space="production")

    z_sand = await store.upsert_entity("Zeus", "Person", "d", e, space="sandbox")
    h_sand = await store.upsert_entity("Hera", "Person", "d", h, space="sandbox")
    await store.add_relation(z_sand, h_sand, "secret_relation", "sandbox only", space="sandbox")

    prod_rels = await store.get_neighbors(z_prod.id, space="production")
    assert [edge.relation_type for edge, _ in prod_rels] == ["married_to"]

    sand_rels = await store.get_neighbors(z_sand.id, space="sandbox")
    assert [edge.relation_type for edge, _ in sand_rels] == ["secret_relation"]


@pytest.mark.asyncio
async def test_reserved_space_rejected_for_writes(rag_factory):
    rag = await rag_factory()
    with pytest.raises(ValueError):
        await rag.store.upsert_entity("Zeus", "Person", "d", fake_embed("zeus"), space="__all__")


@pytest.mark.asyncio
async def test_relation_embeddings_absent_when_disabled(rag_factory):
    """Turning embed_relations off must leave the edge table without a vector
    column, so the cost of the second channel really is opt-out-able."""
    rag = await rag_factory(embed_relations=False)
    rows = await rag.store.client._fetch(
        "SELECT 1 FROM pg_attribute WHERE attrelid = $1::regclass "
        "AND attname = 'embedding' AND NOT attisdropped",
        f'"{rag.config.realm}"."relations"',
    )
    assert not rows
    assert await rag.store.search_similar_relations(fake_embed("married"), top_k=3) == []


@pytest.mark.asyncio
async def test_relation_embeddings_when_enabled(rag_factory):
    rag = await rag_factory(embed_relations=True)
    store = rag.store
    z = await store.upsert_entity("Zeus", "Person", "d", fake_embed("zeus"))
    h = await store.upsert_entity("Hera", "Person", "d", fake_embed("hera"))
    p = await store.upsert_entity("Poseidon", "Person", "d", fake_embed("poseidon"))

    await store.add_relation(z, h, "married_to", "spouse", embedding=fake_embed("zeus hera king"))
    await store.add_relation(z, p, "brother_of", "sibling", embedding=fake_embed("poseidon sea"))

    hits = await store.search_similar_relations(fake_embed("poseidon sea"), top_k=2)
    assert hits and hits[0][0].relation_type == "brother_of"
