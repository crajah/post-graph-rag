"""Tests for concurrent indexing and community re-ranking."""
import asyncio

import pytest

from post_graph_rag import DocumentMetadata, QueryParam, RAGConfig
from post_graph_rag.engine import GraphRAG
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

SHARED = ExtractionResult(
    entities=[
        Entity(name="Charles Babbage", type="Person", description="inventor", aliases=["Babbage"]),
        Entity(name="Analytical Engine", type="Technology", description="mechanical computer"),
    ],
    triples=[Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine")],
)


def _long_text(paragraphs=16):
    return "\n".join(
        f"Charles Babbage designed the Analytical Engine, paragraph {i}, long enough to be kept."
        for i in range(paragraphs)
    )


# ------------------------------------------------------------- concurrency

@pytest.mark.asyncio
async def test_concurrent_indexing_matches_sequential(rag_factory):
    """Parallelism must not change the resulting graph."""
    seq = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=1)
    par = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=4)
    text = _long_text()

    seq_res = await seq.index_text(text, metadata=DocumentMetadata(document="d.txt"))
    par_res = await par.index_text(text, metadata=DocumentMetadata(document="d.txt"))
    assert len(seq_res) == len(par_res) > 1

    async def counts(rag):
        rows = await rag.store.client._fetch(
            f'SELECT (SELECT count(*) FROM "{rag.config.realm}"."entities") e, '
            f'(SELECT count(*) FROM "{rag.config.realm}"."relations") r, '
            f'(SELECT count(*) FROM "{rag.config.realm}"."documents") d')
        return rows[0]["e"], rows[0]["r"], rows[0]["d"]

    assert await counts(seq) == await counts(par)


@pytest.mark.asyncio
async def test_concurrent_indexing_preserves_entity_resolution(rag_factory):
    """The write phase is serialised precisely so concurrent chunks cannot split
    an entity that should have merged."""
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=8)
    await rag.index_text(_long_text(24), metadata=DocumentMetadata(document="d.txt"))

    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."entities" '
        "WHERE lower(payload->>'name') = 'charles babbage'")
    assert rows[0]["n"] == 1


@pytest.mark.asyncio
async def test_prepare_phase_runs_in_parallel(rag_factory):
    """The network-bound phase should overlap, not serialise."""
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=4)
    in_flight, peak = 0, 0
    original = rag.llm.get_embedding

    async def tracked(text):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.02)
            return await original(text)
        finally:
            in_flight -= 1

    rag.llm.get_embedding = tracked
    await rag.index_text(_long_text(), metadata=DocumentMetadata(document="d.txt"))
    assert peak > 1, "chunks did not overlap"


@pytest.mark.asyncio
async def test_context_threaded_across_batches(rag_factory):
    """A chunk should see entities discovered by earlier batches."""
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=2)
    await rag.index_text(_long_text(20), metadata=DocumentMetadata(document="d.txt"))

    later = [c for c, _ in rag.llm.chat_calls[6:]]
    assert any("Charles Babbage" in c[-1]["content"] for c in later)


@pytest.mark.asyncio
async def test_index_documents_accepts_pre_chunked_input(rag_factory):
    rag = await rag_factory(extraction=SHARED, max_concurrent_chunks=3)
    chunks = [(f"Babbage designed engines, part {i}.", DocumentMetadata(document="d.txt", paragraph=i))
              for i in range(1, 6)]
    res = await rag.index_documents(chunks)
    assert len(res) == 5
    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."documents"')
    assert rows[0]["n"] == 5


@pytest.mark.asyncio
async def test_single_concurrency_still_works(rag_factory):
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=1)
    res = await rag.index_text(_long_text(), metadata=DocumentMetadata(document="d.txt"))
    assert len(res) > 1


@pytest.mark.asyncio
async def test_one_failing_chunk_does_not_kill_its_batch(rag_factory):
    """Sequential indexing lost only the offending chunk; concurrency must not
    be worse. A bare asyncio.gather would drop every chunk in the batch."""
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=4)
    calls = {"n": 0}
    original = rag.extractor.extract_from_text

    async def flaky(text, context=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("extraction blew up")
        return await original(text, context=context)

    rag.extractor.extract_from_text = flaky
    res = await rag.index_text(_long_text(), metadata=DocumentMetadata(document="d.txt"))

    # Everything except the one bad chunk survives.
    assert len(res) == calls["n"] - 1
    rows = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."documents"')
    assert rows[0]["n"] == len(res) > 1


@pytest.mark.asyncio
async def test_total_failure_raises_rather_than_reporting_success(rag_factory):
    """Skipping a bad chunk is recovery; skipping every chunk is an outage.

    Returning an empty list for a total failure is indistinguishable from
    successfully indexing an empty document.
    """
    rag = await rag_factory(extraction=SHARED, chunk_chars=200, chunk_overlap_chars=0,
                            max_concurrent_chunks=4)

    async def always_fails(text, context=None):
        raise RuntimeError("endpoint down")

    rag.extractor.extract_from_text = always_fails
    with pytest.raises(RuntimeError, match="endpoint down"):
        await rag.index_text(_long_text(), metadata=DocumentMetadata(document="d.txt"))


def test_transport_errors_are_retryable():
    """Concurrency can momentarily exhaust a router's connection pool; that
    surfaces as a bare connection error with no HTTP status."""
    from post_graph_rag.llm import _is_retryable

    for msg in ["Connection error.", "Server disconnected without sending a response",
                "Connection reset by peer", "[Errno 32] Broken pipe"]:
        assert _is_retryable(Exception(msg)), msg
    # Genuine mistakes must still fail fast rather than burn the retry budget.
    for msg in ["Invalid API key", "model not found", "invalid_request_error"]:
        assert not _is_retryable(Exception(msg)), msg


# --------------------------------------------------- community re-ranking



def _ranker(**cfg):
    rag = GraphRAG.__new__(GraphRAG)
    params = dict(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    params.update(cfg)
    rag.config = RAGConfig(**params)
    return rag


NICHE = {"community_id": "1", "title": "Steampunk alternate histories",
         "summary": "s", "findings": [], "rating": 4.0, "size": 4, "distance": 0.30}
CENTRAL = {"community_id": "2", "title": "Babbage and the engines",
           "summary": "s", "findings": [], "rating": 9.0, "size": 80, "distance": 0.34}


def test_importance_and_size_outrank_a_marginally_closer_niche():
    """The observed failure: a small niche cluster beat the central theme on
    'what are the main themes?' because it sat fractionally closer."""
    ranked = _ranker()._rank_communities([dict(NICHE), dict(CENTRAL)], top_k=2)
    assert ranked[0]["title"] == "Babbage and the engines"


def test_pure_similarity_is_recoverable():
    ranked = _ranker(community_weight_importance=0.0, community_weight_size=0.0
                     )._rank_communities([dict(NICHE), dict(CENTRAL)], top_k=2)
    assert ranked[0]["title"] == "Steampunk alternate histories"


def test_clearly_closer_community_still_wins():
    """Blending must not override a decisive similarity margin."""
    near = dict(NICHE, distance=0.05)
    ranked = _ranker()._rank_communities([near, dict(CENTRAL)], top_k=2)
    assert ranked[0]["title"] == "Steampunk alternate histories"


def test_ranking_truncates_to_top_k():
    items = [dict(CENTRAL, community_id=str(i), distance=0.1 * i) for i in range(6)]
    assert len(_ranker()._rank_communities(items, top_k=3)) == 3


def test_ranking_handles_identical_candidates():
    items = [dict(CENTRAL, community_id=str(i)) for i in range(3)]
    ranked = _ranker()._rank_communities(items, top_k=3)
    assert len(ranked) == 3
    assert all("score" in c for c in ranked)


def test_ranking_handles_single_candidate():
    ranked = _ranker()._rank_communities([dict(CENTRAL)], top_k=5)
    assert len(ranked) == 1


@pytest.mark.asyncio
async def test_global_mode_overfetches_then_reranks(rag_factory):
    rag = await rag_factory(extraction=SHARED)
    await rag.index_document("Babbage designed engines.")
    seen = {}
    original = rag.store.search_similar_communities

    async def spy(vec, top_k=5, space=None, **kw):
        seen["top_k"] = top_k
        return await original(vec, top_k=top_k, space=space, **kw)

    rag.store.search_similar_communities = spy
    await rag.query_data("themes", param=QueryParam(mode="global", top_k=3))
    assert seen["top_k"] > 3, "candidates were not over-fetched before re-ranking"
