"""Importance-scored archiving: demotion, reversible, telemetry-gated.

Archiving borrows dormancy's exclusion machinery, so the assertions mirror
dormancy's: an archived entity leaves retrieval and community builds but
stays in the table, and restore brings it back.
"""
import pytest
from conftest import VOCAB_DIM

pytestmark = pytest.mark.asyncio

EMB = [0.1] * VOCAB_DIM


async def _seed(rag, n=6):
    vs = []
    for i in range(n):
        vs.append(await rag.store.upsert_entity(
            name=f"E{i}", entity_type="Thing", description="d", embedding=EMB))
    # give E0 structure and a query touch so it scores high; leave the rest cold
    await rag.store.add_relation(vs[0], vs[1], "linked", description="l")
    await rag.store.add_relation(vs[0], vs[2], "linked", description="l")
    return vs


class TestTelemetryGate:
    async def test_refuses_without_telemetry(self, rag_factory):
        rag = await rag_factory()                      # flag off
        await _seed(rag)
        with pytest.raises(ValueError, match="record_retrieval_events"):
            await rag.apply_retention()


class TestScoringAndArchive:
    async def test_dry_run_writes_nothing(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed(rag)
        rep = await rag.apply_retention(threshold=1.0, dry_run=True)  # everything below
        assert rep.dry_run and rep.scored == 6
        # nothing archived on disk
        arch = await rag.store.client.find_vertices(
            "entities", realm=rag.config.realm, where=[("archived_at", "not_null", None)])
        assert arch == []

    async def test_archive_marks_and_withholds(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed(rag)
        await rag.query_data("about E0?")              # touch E0 via telemetry
        # threshold 1.0 archives everything scoring below 1.0 -- i.e. the cold
        # entities, but E0 (hit + degree) should score highest
        rep = await rag.apply_retention(threshold=1.0, dry_run=False)
        assert not rep.dry_run and rep.archived >= 1
        arch = await rag.store.client.find_vertices(
            "entities", realm=rag.config.realm, where=[("archived_at", "not_null", None)])
        assert len(arch) == rep.archived
        # archived entities are withheld from vector retrieval
        live = await rag.store.search_similar_entities(EMB, top_k=10)
        live_ids = {v.id for v, _d in live}
        assert not (set(rep.archived_ids) & live_ids)

    async def test_restore_reverses(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed(rag)
        rep = await rag.apply_retention(threshold=1.0, dry_run=False)
        assert rep.archived >= 1
        n = await rag.restore_archived(rep.archived_ids)
        assert n == rep.archived
        arch = await rag.store.client.find_vertices(
            "entities", realm=rag.config.realm, where=[("archived_at", "not_null", None)])
        assert arch == []

    async def test_archived_excluded_from_community_build(self, rag_factory):
        from conftest import FakeLLM

        from post_graph_rag.config import RAGConfig
        from post_graph_rag.reporting import CommunityReport, CommunityReporter, Finding
        rag = await rag_factory(record_retrieval_events=True)
        cfg = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
        rag.reporter = CommunityReporter(FakeLLM(cfg, extraction=CommunityReport(
            title="t", summary="s", findings=[Finding(summary="f", explanation="e")], rating=5.0)))
        await _seed(rag)
        # archive everything, then snapshot must be empty of them
        rep = await rag.apply_retention(threshold=1.0, dry_run=False)
        ents, _rels = await rag.store.graph_snapshot()
        snap_ids = {e["id"] for e in ents}
        assert not (set(rep.archived_ids) & snap_ids)

    async def test_nothing_archived_leaves_graph_intact(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed(rag)
        rep = await rag.apply_retention(threshold=0.0, dry_run=False)  # nothing below 0
        assert rep.archived == 0
        live = await rag.store.search_similar_entities(EMB, top_k=10)
        assert len(live) == 6
