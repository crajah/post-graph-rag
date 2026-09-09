"""Deleting a document must retire what it asserted, and say what it contributed.

Two behaviours that turned out to be one story. Relations were being marked
dormant correctly and then returned by every read path anyway, so to a user
"relations are not getting deleted" -- and relations written before provenance
existed could not be retired at all, because there was nothing recorded to
withdraw. The document-statistics API is the other half: it is how a caller
sees what a document put into the graph before deciding to remove it.
"""
import json

import pytest
from conftest import fake_embed
from post_graph import RESERVED_SPACE_ALL

SPACE = "default"

# Width of the shared realm-column tables, which one test has to match. The
# per-realm schemas the rest of the suite uses are created fresh at the test
# vocabulary's width and are unaffected.
_SHARED_DIM = 1536


def _embed(text, dim=None):
    """Test embedding, zero-padded when a wider shared table demands it."""
    vec = fake_embed(text)
    return vec if dim is None else vec + [0.0] * (dim - len(vec))


async def _index(store, doc_key, entities, relations, space=SPACE, text=None, dim=None):
    """Write one document: a chunk, its entities, its mentions, its relations."""
    body = text or f"document {doc_key} mentioning {' and '.join(entities)}"
    chunk = await store.add_document(
        body, _embed(body, dim),
        metadata={"doc_key": doc_key, "source": doc_key}, space=space)
    verts = {}
    for name in entities:
        v = await store.upsert_entity(name, "Person", f"about {name}", _embed(name, dim), space=space)
        verts[name] = v
        await store.add_doc_mention(chunk, v, space=space)
    for a, rel, b in relations:
        await store.add_relation(verts[a], verts[b], rel, f"{a} {rel} {b}",
                                 space=space, source_chunk=chunk.id)
    # The engine calls this after indexing a document's mentions; revival of a
    # previously dormant entity (and of relations stranded with it) happens
    # there, so a fixture that skips it would not exercise the real path.
    await store.refresh_dormancy([v.id for v in verts.values()], space=space)
    return chunk, verts


async def _relation_row(store, edge_id):
    ref = store.client._get_table_ref("relations", store.realm)
    rows = await store.client._fetch(
        f"SELECT payload FROM {ref} WHERE realm = $1 AND id = $2", store.realm, int(edge_id))
    pl = rows[0]["payload"]
    return pl if isinstance(pl, dict) else json.loads(pl)


# --------------------------------------------------------------- the user bug

@pytest.mark.asyncio
async def test_deleted_document_relations_vanish_from_every_read_path(rag_factory):
    """The reported symptom: deletion marked relations dormant and every read
    path returned them anyway, so nothing appeared to have been deleted."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "doomed.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])

    assert await store.get_neighbors(verts["Zeus"].id, space=SPACE)
    await store.delete_document_chunks("doomed.txt", space=SPACE)

    assert await store.get_neighbors(verts["Zeus"].id, space=SPACE) == []
    assert await store.get_all_relations(space=SPACE) == []
    assert await store.search_relations_text("married", space=SPACE) == []
    assert await store.get_neighborhood(verts["Zeus"].id, max_hops=2, space=SPACE) == []
    _, rels = await store.graph_snapshot(space=SPACE)
    assert rels == []


@pytest.mark.asyncio
async def test_audit_callers_can_still_see_retired_relations(rag_factory):
    """Nothing is deleted -- dormant rows stay reachable on request, which is
    what makes this a retirement rather than a delete."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "doomed.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await store.delete_document_chunks("doomed.txt", space=SPACE)

    back = await store.get_neighbors(verts["Zeus"].id, space=SPACE, include_dormant=True)
    assert [e.relation_type for e, _ in back] == ["married_to"]
    assert await store.search_relations_text("married", space=SPACE, include_dormant=True)
    assert await store.get_all_relations(space=SPACE, include_dormant=True)


@pytest.mark.asyncio
async def test_shared_relation_survives_deleting_one_of_its_documents(rag_factory):
    """Two documents assert the same relation; removing one leaves it current
    with the weight of the evidence that remains."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "first.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await _index(store, "second.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])

    edge = await store.find_relation(verts["Zeus"].id, "married_to", verts["Hera"].id, space=SPACE)
    assert (await _relation_row(store, edge.id))["weight"] == 2

    await store.delete_document_chunks("first.txt", space=SPACE)

    payload = await _relation_row(store, edge.id)
    assert payload.get("dormant_since") is None
    assert payload["weight"] == 1
    assert len(payload["sources"]) == 1
    assert [e.relation_type for e, _ in await store.get_neighbors(verts["Zeus"].id, space=SPACE)] \
        == ["married_to"]


# ------------------------------------------------- relations with no provenance

@pytest.mark.asyncio
async def test_legacy_relation_without_sources_is_retired_by_orphaned_endpoints(rag_factory):
    """Rows written before provenance existed record no sources, so withdrawal
    can never match them. They are reachable through their endpoints instead."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "only.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    edge = await store.find_relation(verts["Zeus"].id, "married_to", verts["Hera"].id, space=SPACE)

    ref = store.client._get_table_ref("relations", store.realm)
    await store.client._execute(
        f"UPDATE {ref} SET payload = payload - 'sources' WHERE realm = $1 AND id = $2",
        store.realm, int(edge.id))
    assert (await _relation_row(store, edge.id)).get("sources") is None

    result = await store.delete_document_chunks("only.txt", space=SPACE)
    assert result["relations_withdrawn"] == 0        # nothing to withdraw
    assert result["relations_orphaned"] == 1

    payload = await _relation_row(store, edge.id)
    assert payload["dormant_reason"] == "orphaned_endpoints"
    assert await store.get_neighbors(verts["Zeus"].id, space=SPACE) == []


@pytest.mark.asyncio
async def test_a_surviving_mention_of_one_endpoint_keeps_the_relation(rag_factory):
    """Both endpoints, not either: a relation to a still-mentioned entity is
    still evidence about that entity."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "only.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await _index(store, "other.txt", ["Zeus"], [])          # Zeus stays mentioned
    edge = await store.find_relation(verts["Zeus"].id, "married_to", verts["Hera"].id, space=SPACE)

    ref = store.client._get_table_ref("relations", store.realm)
    await store.client._execute(
        f"UPDATE {ref} SET payload = payload - 'sources' WHERE realm = $1 AND id = $2",
        store.realm, int(edge.id))

    result = await store.delete_document_chunks("only.txt", space=SPACE)
    assert result["relations_orphaned"] == 0
    assert (await _relation_row(store, edge.id)).get("dormant_since") is None


@pytest.mark.asyncio
async def test_orphaned_relation_revives_when_an_endpoint_is_mentioned_again(rag_factory):
    """Revival symmetry, and only for the endpoint reason -- a relation whose
    sources were withdrawn is dormant for a stronger reason than this."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "only.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    edge = await store.find_relation(verts["Zeus"].id, "married_to", verts["Hera"].id, space=SPACE)
    ref = store.client._get_table_ref("relations", store.realm)
    await store.client._execute(
        f"UPDATE {ref} SET payload = payload - 'sources' WHERE realm = $1 AND id = $2",
        store.realm, int(edge.id))
    await store.delete_document_chunks("only.txt", space=SPACE)
    assert (await _relation_row(store, edge.id))["dormant_reason"] == "orphaned_endpoints"

    await _index(store, "revival.txt", ["Zeus"], [])
    payload = await _relation_row(store, edge.id)
    assert payload.get("dormant_since") is None
    assert payload.get("revived_at")
    assert [e.relation_type for e, _ in await store.get_neighbors(verts["Zeus"].id, space=SPACE)] \
        == ["married_to"]


@pytest.mark.asyncio
async def test_withdrawn_relation_does_not_revive_with_a_neighbouring_entity(rag_factory):
    """The reasons are not interchangeable: a document reappearing that mentions
    an endpoint does not re-assert a relation whose own sources are gone."""
    rag = await rag_factory()
    store = rag.store
    _, verts = await _index(store, "only.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    edge = await store.find_relation(verts["Zeus"].id, "married_to", verts["Hera"].id, space=SPACE)
    await store.delete_document_chunks("only.txt", space=SPACE)
    assert (await _relation_row(store, edge.id))["dormant_reason"] == "sources_withdrawn"

    await _index(store, "revival.txt", ["Zeus"], [])
    assert (await _relation_row(store, edge.id)).get("dormant_since") is not None
    assert await store.get_neighbors(verts["Zeus"].id, space=SPACE) == []


@pytest.mark.asyncio
async def test_sweep_repairs_stranded_rows_and_leaves_live_ones_alone(rag_factory):
    """The one-shot repair an existing deployment runs once."""
    rag = await rag_factory()
    store = rag.store
    _, doomed = await _index(store, "gone.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    _, live = await _index(store, "kept.txt", ["Guido", "Python"], [("Guido", "created", "Python")])
    stranded = await store.find_relation(doomed["Zeus"].id, "married_to", doomed["Hera"].id, space=SPACE)
    survivor = await store.find_relation(live["Guido"].id, "created", live["Python"].id, space=SPACE)

    ref = store.client._get_table_ref("relations", store.realm)
    await store.client._execute(
        f"UPDATE {ref} SET payload = payload - 'sources' WHERE realm = $1", store.realm)
    await store.delete_document_chunks("gone.txt", space=SPACE)
    # Undo the deletion's own orphan pass so the sweep has work to do.
    await store.client._execute(
        f"UPDATE {ref} SET payload = payload - 'dormant_since' - 'dormant_reason' "
        f"WHERE realm = $1 AND id = $2", store.realm, int(stranded.id))

    assert await store.sweep_orphaned_relations(space=SPACE) == 1
    assert (await _relation_row(store, stranded.id))["dormant_reason"] == "orphaned_endpoints"
    assert (await _relation_row(store, survivor.id)).get("dormant_since") is None
    assert await store.sweep_orphaned_relations(space=SPACE) == 0      # idempotent


@pytest.mark.asyncio
async def test_withdrawal_ignores_other_spaces(rag_factory):
    """Space scoping holds through the set-based rewrite."""
    rag = await rag_factory()
    store = rag.store
    await _index(store, "shared-key.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")],
                 space="production")
    _, sand = await _index(store, "shared-key.txt", ["Zeus", "Hera"],
                           [("Zeus", "married_to", "Hera")], space="sandbox")

    await store.delete_document_chunks("shared-key.txt", space="production")

    kept = await store.get_neighbors(sand["Zeus"].id, space="sandbox")
    assert [e.relation_type for e, _ in kept] == ["married_to"]


# ---------------------------------------------------------- document statistics

@pytest.mark.asyncio
async def test_document_stats_counts_what_the_document_contributed(rag_factory):
    rag = await rag_factory()
    store = rag.store
    text = "Zeus and Hera and Cronus appear here"
    await _index(store, "doc-a.txt", ["Zeus", "Hera", "Cronus"],
                 [("Zeus", "married_to", "Hera"), ("Cronus", "father_of", "Zeus")], text=text)

    stats = await store.document_stats("doc-a.txt", space=SPACE)
    assert stats.found is True
    assert stats.chunks == 1
    assert stats.chunk_bytes == len(text)
    assert stats.entities_mentioned == 3
    assert stats.entities_current == 3
    assert stats.entities_dormant == 0
    assert stats.relations == 2
    assert stats.first_indexed_at and stats.last_indexed_at


@pytest.mark.asyncio
async def test_absent_document_is_zeros_not_an_exception(rag_factory):
    rag = await rag_factory()
    stats = await rag.store.document_stats("never-indexed.txt", space=SPACE)
    assert stats.found is False
    assert stats.chunks == 0 and stats.entities_mentioned == 0 and stats.relations == 0


@pytest.mark.asyncio
async def test_batch_stats_equal_the_singles(rag_factory):
    rag = await rag_factory()
    store = rag.store
    await _index(store, "one.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await _index(store, "two.txt", ["Guido", "Python"], [("Guido", "created", "Python")])

    keys = ["one.txt", "two.txt", "absent.txt"]
    batch = await store.documents_stats(keys, space=SPACE)
    assert set(batch) == set(keys)
    for key in keys:
        assert batch[key].to_dict() == (await store.document_stats(key, space=SPACE)).to_dict()


@pytest.mark.asyncio
async def test_dormant_split_moves_when_a_sibling_document_is_deleted(rag_factory):
    """Hera is mentioned by both documents, Cronus by only one. Deleting the
    second leaves Hera current and Cronus dormant, and the split says so."""
    rag = await rag_factory()
    store = rag.store
    await _index(store, "keep.txt", ["Zeus", "Hera"], [])
    await _index(store, "drop.txt", ["Hera", "Cronus"], [])

    assert (await store.document_stats("keep.txt", space=SPACE)).entities_dormant == 0
    await store.delete_document_chunks("drop.txt", space=SPACE)

    keep = await store.document_stats("keep.txt", space=SPACE)
    assert keep.entities_mentioned == 2
    assert keep.entities_current == 2 and keep.entities_dormant == 0

    dropped = await store.document_stats("drop.txt", space=SPACE)
    assert dropped.found is False


@pytest.mark.asyncio
async def test_document_graph_returns_the_contributed_subgraph(rag_factory):
    rag = await rag_factory()
    store = rag.store
    await _index(store, "doc-a.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await _index(store, "doc-b.txt", ["Guido", "Python"], [("Guido", "created", "Python")])

    graph = await store.document_graph("doc-a.txt", space=SPACE)
    assert sorted(e["name"] for e in graph["entities"]) == ["Hera", "Zeus"]
    assert [r["type"] for r in graph["relations"]] == ["married_to"]
    assert graph["relations"][0]["from_name"] == "Zeus"
    assert graph["truncated"] is False


@pytest.mark.asyncio
async def test_document_graph_flags_truncation(rag_factory):
    """Silent truncation reads as completeness, so the cap is reported."""
    rag = await rag_factory()
    store = rag.store
    await _index(store, "wide.txt", ["Zeus", "Hera", "Cronus"], [])

    graph = await store.document_graph("wide.txt", space=SPACE, max_entities=2)
    assert len(graph["entities"]) == 2
    assert graph["entities_truncated"] is True
    assert graph["truncated"] is True


@pytest.mark.asyncio
async def test_per_document_reads_refuse_the_cross_space_wildcard(rag_factory):
    """RAGConfig.space always resolves, so the reachable cross-tenant read is an
    explicit __all__ -- which these methods refuse rather than honour."""
    rag = await rag_factory()
    with pytest.raises(ValueError):
        await rag.store.document_stats("anything.txt", space=RESERVED_SPACE_ALL)
    with pytest.raises(ValueError):
        await rag.store.documents_stats(["anything.txt"], space=RESERVED_SPACE_ALL)
    with pytest.raises(ValueError):
        await rag.store.document_graph("anything.txt", space=RESERVED_SPACE_ALL)


@pytest.mark.asyncio
async def test_lifecycle_holds_in_realm_column_mode(rag_factory):
    """Everything above runs under schema_per_realm; the other mode shares
    tables across realms and must behave identically."""
    # Realm-column mode shares one set of tables across every realm, so the
    # embedding width is whatever those tables already have rather than the
    # test vocabulary's.
    rag = await rag_factory(schema_per_realm=False, embedding_dim=_SHARED_DIM)
    store = rag.store
    _, verts = await _index(store, "doomed.txt", ["Zeus", "Hera"],
                            [("Zeus", "married_to", "Hera")], dim=_SHARED_DIM)

    stats = await store.document_stats("doomed.txt", space=SPACE)
    assert stats.chunks == 1 and stats.relations == 1

    await store.delete_document_chunks("doomed.txt", space=SPACE)
    assert await store.get_neighbors(verts["Zeus"].id, space=SPACE) == []
    assert (await store.document_stats("doomed.txt", space=SPACE)).found is False


# ------------------------------------------------------ the whole way through

@pytest.mark.asyncio
async def test_deleted_document_leaves_the_query_pipeline_with_nothing(rag_factory):
    """The end-to-end statement of the bug: index through the engine, delete
    through the engine, and the retrieval pipeline must have no relations left
    to answer from. Every read path is exercised in one call here, since
    query_data fuses traversal, relation similarity and lexical search."""
    from post_graph_rag import DocumentMetadata, QueryParam
    from post_graph_rag.extractor import Entity, ExtractionResult, Triple

    extraction = ExtractionResult(
        entities=[
            Entity(name="Zeus", type="Person", description="king of the Olympian gods"),
            Entity(name="Hera", type="Person", description="wife of Zeus"),
        ],
        triples=[Triple(subject="Zeus", predicate="married_to", object="Hera",
                        description="spouse")],
    )
    rag = await rag_factory(extraction=extraction)
    await rag.index_document("Zeus is king of the Olympian gods, married to Hera.",
                             metadata=DocumentMetadata(document="zeus.pdf"))

    stats = await rag.document_stats("zeus.pdf")
    assert stats.found and stats.relations == 1 and stats.entities_mentioned == 2

    before = await rag.query_data("Who is Zeus married to?", QueryParam(mode="mix"))
    assert before["data"]["relationships"], "fixture is not exercising the relation path"

    await rag.store.delete_document_chunks("zeus.pdf", space=rag.store.space)

    after = await rag.query_data("Who is Zeus married to?", QueryParam(mode="mix"))
    assert after["data"]["relationships"] == []
    assert after["data"]["chunks"] == []
    assert (await rag.document_stats("zeus.pdf")).found is False


@pytest.mark.asyncio
async def test_engine_exposes_the_document_apis(rag_factory):
    """The registry talks to GraphRAG, not to the store."""
    rag = await rag_factory()
    store = rag.store
    await _index(store, "a.txt", ["Zeus", "Hera"], [("Zeus", "married_to", "Hera")])
    await _index(store, "b.txt", ["Guido"], [])

    batch = await rag.documents_stats(["a.txt", "b.txt"])
    assert batch["a.txt"].relations == 1 and batch["b.txt"].relations == 0
    graph = await rag.document_graph("a.txt")
    assert [r["type"] for r in graph["relations"]] == ["married_to"]
    assert await rag.sweep_orphaned_relations() == 0
