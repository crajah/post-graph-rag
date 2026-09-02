"""Coverage telemetry and least-explored queries.

The flag-off default is the most important assertion: a non-telemetry realm
must gain no table and pay no write. Everything else is the poll-work loop a
breadth-first consumer runs.
"""
import pytest

from post_graph_rag.config import RAGConfig
from post_graph_rag.reporting import CommunityReport, CommunityReporter, Finding

from conftest import FakeLLM

pytestmark = pytest.mark.asyncio

REPORT = CommunityReport(
    title="A cluster", summary="s",
    findings=[Finding(summary="f", explanation="e")], rating=7.0)


def _fake_reporter():
    cfg = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    return CommunityReporter(FakeLLM(cfg, extraction=REPORT))

from conftest import VOCAB_DIM

EMB = [0.1] * VOCAB_DIM


async def _seed_graph(rag):
    a = await rag.store.upsert_entity(name="Ada", entity_type="Person",
                                      description="d", embedding=EMB)
    b = await rag.store.upsert_entity(name="Babbage", entity_type="Person",
                                      description="d", embedding=EMB)
    c = await rag.store.upsert_entity(name="Engine", entity_type="Thing",
                                      description="d", embedding=EMB)
    await rag.store.add_relation(a, b, "worked_with", description="w")
    await rag.store.add_relation(b, c, "designed", description="d")
    return a, b, c


class TestFlagOff:
    async def test_no_table_no_writes_by_default(self, rag_factory):
        rag = await rag_factory()
        assert rag.config.record_retrieval_events is False
        await _seed_graph(rag)
        await rag.query_data("who worked with Ada?")
        ref = rag.store.client._get_table_ref("retrieval_events", rag.config.realm)
        rows = await rag.store.client._fetch(
            f"SELECT to_regclass('{ref}') AS t")
        assert rows[0]["t"] is None            # table never created


class TestFlagOn:
    async def test_event_written_per_query(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed_graph(rag)
        await rag.query_data("who worked with Ada?")
        await rag.query_data("what did Babbage design?")
        n = await rag.store.client.count_vertices(
            "retrieval_events", realm=rag.config.realm)
        assert n == 2

    async def test_query_text_never_stored(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed_graph(rag)
        secret = "extremely confidential question about Ada"
        await rag.query_data(secret)
        evs = await rag.store.client.find_vertices(
            "retrieval_events", realm=rag.config.realm)
        assert len(evs) == 1
        assert "confidential" not in str(evs[0].payload)
        assert len(evs[0].payload["query_sha256"]) == 64

    async def test_poisoned_write_does_not_fail_query(self, rag_factory, monkeypatch):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed_graph(rag)

        async def boom(*a, **k):
            raise RuntimeError("telemetry backend down")
        monkeypatch.setattr(rag.store.client, "add_vertex", boom)
        out = await rag.query_data("who worked with Ada?")   # must not raise
        assert out["status"] == "success"

    async def test_purge_retention(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed_graph(rag)
        await rag.query_data("who worked with Ada?")
        purged = await rag.purge_retrieval_events(before="2000-01-01T00:00:00")
        assert purged == 0                                    # nothing that old
        purged = await rag.purge_retrieval_events(before="2999-01-01T00:00:00")
        assert purged == 1
        assert await rag.store.client.count_vertices(
            "retrieval_events", realm=rag.config.realm) == 0


class TestCoverageQueries:
    async def test_coverage_and_least_explored(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        rag.reporter = _fake_reporter()
        await _seed_graph(rag)
        built = await rag.build_communities()
        assert built.get("communities", 0) >= 1
        await rag.query_data("who worked with Ada?")
        cov = await rag.coverage()
        assert cov, "no communities in coverage"
        assert all(hasattr(c, "hit_share") for c in cov)
        # least-explored ordering: ascending hits
        hits = [c.retrieval_hits for c in cov]
        assert hits == sorted(hits)
        top = await rag.least_explored_communities(k=1)
        assert len(top) == 1

    async def test_dark_entities_shrink_after_query(self, rag_factory):
        rag = await rag_factory(record_retrieval_events=True)
        await _seed_graph(rag)
        before = await rag.dark_entities()
        assert len(before) == 3                               # nobody touched yet
        await rag.query_data("who worked with Ada?")
        after = await rag.dark_entities()
        assert len(after) < 3
