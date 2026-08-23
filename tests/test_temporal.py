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

def test_document_key_uses_both_parts():
    assert document_key("https://x/a", "A.txt") == "https://x/a::A.txt"
    assert document_key(None, "A.txt") == "A.txt"
    assert document_key("https://x/a", None) == "https://x/a"
    assert document_key(None, None) == "unkeyed"


def test_document_key_does_not_collapse_on_a_constant_source():
    """The failure this guards against deleted 15 of every 16 documents.

    A caller passing a corpus name as the source — rather than a path — used to
    produce one key for the whole corpus. Since a matching key means re-index,
    every document removed the one before it.
    """
    keys = {document_key("ect", f"WDC-2023-q{q}") for q in range(1, 5)}
    assert len(keys) == 4


def test_document_key_ignores_surrounding_whitespace():
    assert document_key(" s://a ", " A.txt ") == "s://a::A.txt"


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
async def test_documents_sharing_a_source_do_not_replace_each_other(rag_factory):
    """Distinct documents under one source must coexist.

    This is the end-to-end shape of a real failure: an ECT-QA run passed
    ``source="ect"`` for all 80 transcripts, every one keyed to "ect", and since
    a matching key means re-index, each transcript deleted the previous one. The
    graph ended with one quarter per company and 92% of relations dormant, and
    the only symptom was the system answering "unanswerable" to almost
    everything — which it was right to do, given what was left.
    """
    rag = await rag_factory(extraction=FRIENDS, chunk_chars=200, chunk_overlap_chars=0,
                            skip_unchanged_documents=False)
    for quarter in ("q1", "q2", "q3"):
        await rag.index_text(
            _long(word="Alice"),
            metadata=DocumentMetadata(source="ect", document=f"ACME-2023-{quarter}"))

    for quarter in ("q1", "q2", "q3"):
        chunks = await rag.store.find_document_chunks(
            document_key("ect", f"ACME-2023-{quarter}"))
        assert chunks, f"{quarter} was deleted by a later document sharing its source"


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
async def _index_friends_then_rivals(rag):
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ExtractionResult(
        entities=[Entity(name="Alice", type="Person", description="d")],
        triples=[Triple(subject="Alice", predicate="rivals_with", object="Bob")],
    )
    await rag.index_document("rivals", metadata=DocumentMetadata(document="b.txt"))
    res = await rag.query_data("Alice", param=QueryParam(mode="mix", top_k=5, ll_keywords=["Alice"]))
    return res["data"]["relationships"]


@pytest.mark.asyncio
async def test_relations_carry_assertion_time_and_sort_newest_first(rag_factory):
    """Recency ordering is a property of the traversal channel, so this pins the
    quota to zero: with both channels live the head of the list is shared."""
    rag = await rag_factory(extraction=FRIENDS, relation_seed_quota=0.0)
    rels = await _index_friends_then_rivals(rag)
    assert all(r["asserted_at"] for r in rels), "assertion time not surfaced"
    assert rels[0]["relation_type"] == "rivals_with", "newest assertion did not lead"


@pytest.mark.asyncio
async def test_assertion_time_survives_the_channel_merge(rag_factory):
    """The similarity channel may take the leading slot, but every relation it
    contributes still carries the provenance the traversal channel attaches —
    otherwise merging would silently strip temporal filtering downstream."""
    rag = await rag_factory(extraction=FRIENDS)
    rels = await _index_friends_then_rivals(rag)
    assert all(r["asserted_at"] for r in rels), "assertion time lost on merge"
    assert {r["relation_type"] for r in rels} == {"friend_of", "rivals_with"}


# --------------------------------------------------------------- hop ordering

def _rag_for_ranking():
    """A GraphRAG whose store is never touched — only ranking is under test."""
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    return GraphRAG(config, llm=FakeLLM(config))


def _triple(name, hops, asserted, weight=1, **kw):
    return {"src_id": "A", "tgt_id": name, "relation_type": "r", "description": "",
            "weight": weight, "negated": False, "confidence": 1.0,
            "valid_from": None, "valid_to": None, "superseded_by": None,
            "asserted_at": asserted, "hops": hops, **kw}


def test_nearer_relations_rank_before_distant_ones():
    """Hop distance must dominate assertion time.

    Assertion time across a corpus is close to arbitrary, so without this a
    three-hop edge that happened to be indexed last would displace an adjacent
    one when the relation token budget truncates.
    """
    rag = _rag_for_ranking()
    triples = [
        _triple("far-but-recent", hops=3, asserted="2026-01-01"),
        _triple("near-but-old", hops=1, asserted="2020-01-01"),
        _triple("mid", hops=2, asserted="2025-01-01"),
    ]
    ranked = rag._filter_temporal(triples, QueryParam())
    assert [t["tgt_id"] for t in ranked] == ["near-but-old", "mid", "far-but-recent"]


def test_newest_first_still_holds_within_one_hop_level():
    """Hop ordering must not destroy the newest-first rule it wraps."""
    rag = _rag_for_ranking()
    triples = [
        _triple("older", hops=1, asserted="2020-01-01"),
        _triple("newer", hops=1, asserted="2024-01-01"),
    ]
    ranked = rag._filter_temporal(triples, QueryParam())
    assert [t["tgt_id"] for t in ranked] == ["newer", "older"]


def test_triples_without_hops_are_treated_as_adjacent():
    """Relations from paths that never recorded a hop must not sort last."""
    rag = _rag_for_ranking()
    triples = [_triple("far", hops=2, asserted="2026-01-01")]
    no_hops = _triple("unknown", hops=1, asserted="2020-01-01")
    del no_hops["hops"]
    ranked = rag._filter_temporal(triples + [no_hops], QueryParam())
    assert ranked[0]["tgt_id"] == "unknown"


# ------------------------------------------------------------ channel quota

def _t(name):
    return {"src_id": "A", "tgt_id": name, "relation_type": "r"}


def test_quota_interleaves_both_channels():
    """Half the slots go to relation search, half to traversal."""
    merged = GraphRAG._merge_by_quota(
        [_t(f"trav{i}") for i in range(4)],
        [_t(f"seed{i}") for i in range(4)],
        0.5,
    )
    names = [t["tgt_id"] for t in merged]
    assert sorted(names) == sorted([f"trav{i}" for i in range(4)] + [f"seed{i}" for i in range(4)])
    # Neither channel may monopolise the front of the list.
    assert len([n for n in names[:4] if n.startswith("seed")]) == 2


def test_quota_zero_keeps_traversal_only_order():
    """quota=0 must leave the shipped behaviour untouched."""
    trav = [_t("a"), _t("b")]
    merged = GraphRAG._merge_by_quota(trav, [_t("s1"), _t("s2")], 0.0)
    assert [t["tgt_id"] for t in merged][:2] == ["a", "b"]


def test_merge_is_inert_when_a_channel_is_empty():
    """Relation embeddings are optional; with none, retrieval is unchanged."""
    trav = [_t("a"), _t("b")]
    assert GraphRAG._merge_by_quota(trav, [], 0.5) == trav
    assert GraphRAG._merge_by_quota([], trav, 0.5) == trav


def test_merge_loses_nothing_when_channels_differ_in_length():
    """A short channel must not truncate the long one."""
    merged = GraphRAG._merge_by_quota([_t(f"t{i}") for i in range(5)], [_t("s0")], 0.5)
    assert len(merged) == 6


def test_relation_channel_keeps_similarity_order():
    """Relation search returns by similarity; re-sorting would discard it."""
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    rag = GraphRAG(config, llm=FakeLLM(config))
    seeded = [
        {**_t("closest"), "asserted_at": "2020-01-01", "hops": 1},
        {**_t("next"), "asserted_at": "2026-01-01", "hops": 1},
    ]
    kept = rag._filter_temporal(seeded, QueryParam(), sort=False)
    assert [t["tgt_id"] for t in kept] == ["closest", "next"]


# ------------------------------------------------------------------- RRF

def _t3(src, pred, tgt):
    return {"src_id": src, "relation_type": pred, "tgt_id": tgt}


def test_rrf_rewards_agreement_between_channels():
    """A triple ranked by two channels beats one ranked first by only one.

    This is the property a quota cannot express: the quota allocates slots by
    a fixed share, so a triple every channel agrees on still waits its turn.
    """
    a = [_t3("A", "r", "1"), _t3("A", "r", "2")]
    b = [_t3("A", "r", "3"), _t3("A", "r", "2")]
    merged = GraphRAG._merge_by_rrf([a, b])
    assert merged[0]["tgt_id"] == "2", "the agreed triple did not lead"


def test_rrf_deduplicates_across_channels():
    a = [_t3("A", "r", "1")]
    b = [_t3("a", "R", "1")]          # same triple, different casing
    assert len(GraphRAG._merge_by_rrf([a, b])) == 1


def test_rrf_loses_nothing():
    a = [_t3("A", "r", str(i)) for i in range(5)]
    b = [_t3("B", "r", str(i)) for i in range(3)]
    assert len(GraphRAG._merge_by_rrf([a, b])) == 8


def test_rrf_with_one_channel_preserves_its_order():
    a = [_t3("A", "r", str(i)) for i in range(4)]
    assert [t["tgt_id"] for t in GraphRAG._merge_by_rrf([a])] == ["0", "1", "2", "3"]


def test_rrf_is_inert_on_empty_channels():
    assert GraphRAG._merge_by_rrf([[], []]) == []
