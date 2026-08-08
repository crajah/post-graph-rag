"""Tests for temporal evolution: re-indexing, dormancy, supersession, validity."""
import pytest

from post_graph_rag import DocumentMetadata, QueryParam, RAGConfig
from post_graph_rag.engine import GraphRAG, _valid_at
from post_graph_rag.extractor import (
    Entity, ExtractionResult, GraphExtractor, Triple, _clean_date, date_sort_key,
)
from post_graph_rag.models import content_hash, document_key

from conftest import FakeLLM

FRIENDS = ExtractionResult(
    entities=[Entity(name="Alice", type="Person", description="colleague"),
              Entity(name="Bob", type="Person", description="colleague")],
    triples=[Triple(subject="Alice", predicate="friend_of", object="Bob",
                    description="close friends")],
)
ENEMIES = ExtractionResult(
    entities=[Entity(name="Alice", type="Person", description="former friend"),
              Entity(name="Bob", type="Person", description="former friend")],
    triples=[Triple(subject="Alice", predicate="enemy_of", object="Bob",
                    description="fell out")],
)
EXCLUSIVE = [{"friend_of", "enemy_of"}]


# ------------------------------------------------------------- date handling

def test_clean_date_accepts_partial_and_rejects_vague():
    assert _clean_date("1625") == "1625"
    assert _clean_date("1625-6-2") == "1625-06-02"
    for vague in ["later", "in his youth", "", None, "unknown"]:
        assert _clean_date(vague) is None, vague


def test_date_sort_key_orders_partial_dates():
    assert date_sort_key("1625") < date_sort_key("1625-06-01")
    assert date_sort_key("1625-12") > date_sort_key("1625-06-30")


def test_undated_relations_are_valid_at_every_date():
    """The core rule: silence about a period means the fact held throughout."""
    for as_of in ["1600", "1625-06-12", "2026"]:
        assert _valid_at(None, None, as_of), as_of


def test_validity_bounds():
    assert _valid_at("1625", "1630", "1627")
    assert not _valid_at("1625", "1630", "1624")
    assert not _valid_at("1625", "1630", "1631")
    assert _valid_at("1625", None, "2026")      # open-ended
    assert _valid_at(None, "1630", "1600")      # unbounded start


# ------------------------------------------------------- validity extraction

def _extractor(extraction=None, **kw):
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    kw.setdefault("gleaning_passes", 0)
    return GraphExtractor(FakeLLM(config, extraction=extraction), **kw)


@pytest.mark.asyncio
async def test_stated_validity_is_kept():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Person", description="d")],
        triples=[Triple(subject="A", predicate="employed_by", object="B",
                        valid_from="1828", valid_to="1839")],
    )
    out = await _extractor(result).extract_from_text("text")
    assert (out.triples[0].valid_from, out.triples[0].valid_to) == ("1828", "1839")


@pytest.mark.asyncio
async def test_absent_validity_stays_absent():
    """A relation with no stated period must not acquire one."""
    result = ExtractionResult(
        entities=[Entity(name="A", type="Person", description="d")],
        triples=[Triple(subject="A", predicate="friend_of", object="B")],
    )
    out = await _extractor(result).extract_from_text("text")
    assert out.triples[0].valid_from is None
    assert out.triples[0].valid_to is None


@pytest.mark.asyncio
async def test_vague_dates_are_discarded_not_coerced():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Person", description="d")],
        triples=[Triple(subject="A", predicate="met", object="B", valid_from="much later")],
    )
    out = await _extractor(result).extract_from_text("text")
    assert out.triples[0].valid_from is None


@pytest.mark.asyncio
async def test_validity_extraction_can_be_disabled():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Person", description="d")],
        triples=[Triple(subject="A", predicate="employed_by", object="B", valid_from="1828")],
    )
    out = await _extractor(result, extract_validity=False).extract_from_text("text")
    assert out.triples[0].valid_from is None
    assert "Do not populate valid_from" in _extractor(result, extract_validity=False).system_prompt


# ---------------------------------------------------------- document identity

def test_document_key_prefers_source():
    assert document_key("https://x/a", "A.txt") == "https://x/a"
    assert document_key(None, "A.txt") == "A.txt"
    assert document_key(None, None) == "unkeyed"


def test_content_hash_detects_change():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_identity_fields_are_typed_and_round_trip():
    """doc_key and content_hash are first-class fields, not `extra` entries, so
    callers can read and compare them directly."""
    meta = DocumentMetadata(document="a.txt", source="s://a")
    meta.doc_key = "s://a"
    meta.content_hash = content_hash("hello")

    restored = DocumentMetadata.from_dict(meta.to_dict())
    assert restored.doc_key == "s://a"
    assert restored.content_hash == content_hash("hello")
    assert restored.extra == {}, "identity leaked into extra instead of typed fields"


@pytest.mark.asyncio
async def test_indexed_chunks_carry_a_comparable_hash(rag_factory):
    """Every stored chunk records its own hash, so two runs can be diffed."""
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    text = _long()
    await rag.index_text(text, metadata=DocumentMetadata(source="s://doc", document="doc.txt"))

    stored = await rag.store.find_document_chunks(document_key("s://doc", "doc.txt"))
    assert stored and all(c["content_hash"] for c in stored)
    assert [c["content_hash"] for c in stored] == [content_hash(c) for c in rag.chunker(text)]


# --------------------------------------------------------------- re-indexing

def _long(n=8, word="Alice"):
    return "\n".join(f"{word} and Bob appear together in paragraph {i}, at length." for i in range(n))


@pytest.mark.asyncio
async def test_reindex_replaces_instead_of_duplicating(rag_factory):
    """Measured before the fix: indexing the same document three times gave three
    document vertices and inflated relation weight to 3."""
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0,
                            skip_unchanged_documents=False)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    for _ in range(3):
        await rag.index_text(_long(), metadata=meta)

    docs = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."documents"')
    rels = await rag.store.client._fetch(
        f"SELECT payload->>'weight' AS w FROM \"{rag.config.realm}\".\"relations\"")
    first_pass = len(rag.chunker(_long()))
    assert docs[0]["n"] == first_pass, "re-indexing duplicated chunks"
    assert int(rels[0]["w"]) <= first_pass, "re-indexing inflated corroboration weight"


@pytest.mark.asyncio
async def test_unchanged_document_is_skipped(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    first = await rag.index_text(_long(), metadata=meta)
    second = await rag.index_text(_long(), metadata=meta)
    assert first and second == [], "unchanged content was re-extracted"


@pytest.mark.asyncio
async def test_changed_document_is_reindexed(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    await rag.index_text(_long(), metadata=meta)
    second = await rag.index_text(_long(n=10), metadata=meta)
    assert second, "changed content was skipped"
    docs = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."documents"')
    assert docs[0]["n"] == len(second)


# ----------------------------------------------------------------- dormancy

@pytest.mark.asyncio
async def test_orphaned_entities_go_dormant_not_deleted(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    await rag.index_text(_long(), metadata=meta)

    removed = await rag.store.delete_document_chunks(document_key("s://doc", "doc.txt"))
    assert removed["chunks"] > 0
    assert removed["dormant"] > 0

    rows = await rag.store.client._fetch(
        f"SELECT payload->>'dormant_since' AS d FROM \"{rag.config.realm}\".\"entities\"")
    assert rows, "entities were deleted rather than marked dormant"
    assert all(r["d"] for r in rows)


@pytest.mark.asyncio
async def test_dormant_entities_are_excluded_from_retrieval(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    await rag.index_text(_long(), metadata=meta)
    await rag.store.delete_document_chunks(document_key("s://doc", "doc.txt"))

    res = await rag.query_data("Alice", param=QueryParam(mode="mix", top_k=5, ll_keywords=["Alice"]))
    assert res["data"]["entities"] == []


@pytest.mark.asyncio
async def test_dormant_entity_revives_when_mentioned_again(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    await rag.index_text(_long(), metadata=meta)
    await rag.store.delete_document_chunks(document_key("s://doc", "doc.txt"))
    await rag.index_text(_long(), metadata=meta)

    rows = await rag.store.client._fetch(
        f"SELECT payload->>'dormant_since' AS d FROM \"{rag.config.realm}\".\"entities\"")
    assert all(r["d"] is None for r in rows), "entity stayed dormant after re-mention"


@pytest.mark.asyncio
async def test_dormant_entities_excluded_from_community_builds(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0)
    meta = DocumentMetadata(source="s://doc", document="doc.txt")
    await rag.index_text(_long(), metadata=meta)
    await rag.store.delete_document_chunks(document_key("s://doc", "doc.txt"))

    entities, _ = await rag.store.graph_snapshot()
    assert entities == [], "dormant entities would cluster into ghost communities"


# ------------------------------------------------------------- supersession

@pytest.mark.asyncio
async def test_later_assertion_supersedes_the_earlier(rag_factory):
    """Alice and Bob are friends, then enemies. Both must not read as current."""
    rag = await rag_factory(extraction=FRIENDS, exclusive_predicate_groups=EXCLUSIVE)
    await rag.index_document("In 2015 Alice and Bob became friends.",
                             metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ENEMIES
    res = await rag.index_document("In 2023 Alice and Bob became enemies.",
                                   metadata=DocumentMetadata(document="b.txt"))
    assert res["relations_superseded"] == 1

    rows = await rag.store.client._fetch(
        f"SELECT relation_type, payload->>'superseded_by' AS s "
        f"FROM \"{rag.config.realm}\".\"relations\" ORDER BY id")
    assert rows[0]["relation_type"] == "friend_of" and rows[0]["s"] is not None
    assert rows[1]["relation_type"] == "enemy_of" and rows[1]["s"] is None


@pytest.mark.asyncio
async def test_superseded_relation_hidden_from_retrieval(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, exclusive_predicate_groups=EXCLUSIVE)
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ENEMIES
    await rag.index_document("enemies", metadata=DocumentMetadata(document="b.txt"))

    res = await rag.query_data("Alice and Bob", param=QueryParam(
        mode="mix", top_k=5, ll_keywords=["Alice"]))
    kinds = {r["relation_type"] for r in res["data"]["relationships"]}
    assert "enemy_of" in kinds
    assert "friend_of" not in kinds, "a superseded fact was presented as current"


@pytest.mark.asyncio
async def test_superseded_relation_recoverable_on_request(rag_factory):
    rag = await rag_factory(extraction=FRIENDS, exclusive_predicate_groups=EXCLUSIVE)
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ENEMIES
    await rag.index_document("enemies", metadata=DocumentMetadata(document="b.txt"))

    res = await rag.query_data("Alice and Bob", param=QueryParam(
        mode="mix", top_k=5, ll_keywords=["Alice"], include_superseded=True))
    kinds = {r["relation_type"] for r in res["data"]["relationships"]}
    assert {"friend_of", "enemy_of"} <= kinds, "history was lost, not just hidden"


@pytest.mark.asyncio
async def test_unrelated_predicates_are_not_superseded(rag_factory):
    """Only predicates declared mutually exclusive supersede each other."""
    rag = await rag_factory(extraction=FRIENDS, exclusive_predicate_groups=EXCLUSIVE)
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ExtractionResult(
        entities=[Entity(name="Alice", type="Person", description="d")],
        triples=[Triple(subject="Alice", predicate="worked_with", object="Bob")],
    )
    res = await rag.index_document("colleagues", metadata=DocumentMetadata(document="b.txt"))
    assert res["relations_superseded"] == 0


@pytest.mark.asyncio
async def test_no_supersession_without_configured_groups(rag_factory):
    rag = await rag_factory(extraction=FRIENDS)
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ENEMIES
    res = await rag.index_document("enemies", metadata=DocumentMetadata(document="b.txt"))
    assert res["relations_superseded"] == 0


# --------------------------------------------------------- as-of retrieval

DATED = ExtractionResult(
    entities=[Entity(name="Alice", type="Person", description="d"),
              Entity(name="Bob", type="Person", description="d")],
    triples=[
        Triple(subject="Alice", predicate="employed_by", object="Bob",
               valid_from="1820", valid_to="1830"),
        Triple(subject="Alice", predicate="knows", object="Bob"),   # undated
    ],
)


@pytest.mark.asyncio
async def test_as_of_filters_dated_but_keeps_undated(rag_factory):
    """The requirement: documents without validity are unaffected."""
    rag = await rag_factory(extraction=DATED)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))

    inside = await rag.query_data("Alice", param=QueryParam(
        mode="mix", top_k=5, ll_keywords=["Alice"], as_of="1825"))
    outside = await rag.query_data("Alice", param=QueryParam(
        mode="mix", top_k=5, ll_keywords=["Alice"], as_of="1900"))

    assert {r["relation_type"] for r in inside["data"]["relationships"]} == {"employed_by", "knows"}
    # Outside the stated period the dated relation drops; the undated one remains.
    assert {r["relation_type"] for r in outside["data"]["relationships"]} == {"knows"}


@pytest.mark.asyncio
async def test_no_as_of_returns_everything(rag_factory):
    rag = await rag_factory(extraction=DATED)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))
    res = await rag.query_data("Alice", param=QueryParam(mode="mix", top_k=5, ll_keywords=["Alice"]))
    assert {r["relation_type"] for r in res["data"]["relationships"]} == {"employed_by", "knows"}


# ----------------------------------------------------------------- recency

@pytest.mark.asyncio
async def test_relations_carry_assertion_time_and_sort_newest_first(rag_factory):
    rag = await rag_factory(extraction=FRIENDS)
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ExtractionResult(
        entities=[Entity(name="Alice", type="Person", description="d")],
        triples=[Triple(subject="Alice", predicate="rivals_with", object="Bob")],
    )
    await rag.index_document("rivals", metadata=DocumentMetadata(document="b.txt"))

    res = await rag.query_data("Alice", param=QueryParam(mode="mix", top_k=5, ll_keywords=["Alice"]))
    rels = res["data"]["relationships"]
    assert all(r["asserted_at"] for r in rels), "assertion time not surfaced"
    assert rels[0]["relation_type"] == "rivals_with", "newest assertion did not lead"
