"""Hierarchical communities: recursive supergraph clustering above L0.

The two invariants that matter: default community_levels=1 produces byte-level
current behaviour (no children, no parents), and when the hierarchy is on it
genuinely nests -- every child has exactly one parent edge.
"""
import pytest

from post_graph_rag.config import RAGConfig
from post_graph_rag.reporting import CommunityReport, CommunityReporter, Finding

from conftest import FakeLLM, VOCAB_DIM

pytestmark = pytest.mark.asyncio

EMB = [0.1] * VOCAB_DIM

REPORT = CommunityReport(
    title="Cluster report", summary="s",
    findings=[Finding(summary="f", explanation="e")], rating=6.0)


def _fake_reporter(rating=6.0):
    cfg = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    rep = CommunityReport(title="Cluster report", summary="s",
                          findings=[Finding(summary="f", explanation="e")],
                          rating=rating)
    return CommunityReporter(FakeLLM(cfg, extraction=rep))


async def _two_cluster_graph(rag):
    """Two dense triangles joined by one weak bridge: L0 finds two clusters,
    L1 should merge them under a single parent."""
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    vs = {}
    for n in names:
        vs[n] = await rag.store.upsert_entity(
            name=n, entity_type="Thing", description="d", embedding=EMB)
    tri = [("A1", "A2"), ("A2", "A3"), ("A1", "A3"),
           ("B1", "B2"), ("B2", "B3"), ("B1", "B3")]
    for a, b in tri:
        await rag.store.add_relation(vs[a], vs[b], "linked", description="l")
    await rag.store.add_relation(vs["A1"], vs["B1"], "bridge", description="b")
    return vs


class TestDefaultUnchanged:
    async def test_levels_1_builds_no_hierarchy(self, rag_factory):
        rag = await rag_factory()
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        res = await rag.build_communities()
        assert res["levels"] == {0: res["communities"]}
        assert await rag.store.communities_at_level(1) == []
        edges = await rag.store.client.find_edges(
            "community_children", realm=rag.config.realm, filters={})
        assert edges == []


class TestHierarchy:
    async def test_two_levels_nest(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        res = await rag.build_communities()
        assert res["levels"].get(0, 0) == 2, res
        assert res["levels"].get(1, 0) == 1, res

        parents = await rag.store.communities_at_level(1)
        assert len(parents) == 1
        kids = await rag.store.community_children(parents[0].id)
        assert len(kids) == 2

        # nesting invariant: each L0 community has exactly one parent edge
        edges = await rag.store.client.find_edges(
            "community_children", realm=rag.config.realm, filters={})
        child_ids = [e.to_id for e in edges]
        assert len(child_ids) == len(set(child_ids)) == 2

    async def test_parent_rating_is_max_of_children(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter(rating=8.5)
        await _two_cluster_graph(rag)
        await rag.build_communities()
        parents = await rag.store.communities_at_level(1)
        assert parents[0].payload["rating"] == 8.5

    async def test_rebuild_clears_all_levels(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        await rag.build_communities()
        await rag.build_communities()          # second build replaces, not adds
        assert len(await rag.store.communities_at_level(0)) == 2
        assert len(await rag.store.communities_at_level(1)) == 1
        edges = await rag.store.client.find_edges(
            "community_children", realm=rag.config.realm, filters={})
        assert len(edges) == 2

    async def test_tree_api(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        await rag.build_communities()
        tree = await rag.get_community_tree()
        assert tree["levels"] == 2
        assert len(tree["roots"]) == 1
        root = tree["roots"][0]
        assert root["level"] == 1
        assert len(root["children"]) == 2
        assert all(c["level"] == 0 for c in root["children"])
        kids = await rag.children_of(root["community_id"])
        assert len(kids) == 2

    async def test_determinism_across_builds(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        r1 = await rag.build_communities()
        r2 = await rag.build_communities()
        assert r1["levels"] == r2["levels"]


class TestLevelFilteredRetrieval:
    async def test_query_param_level_filter(self, rag_factory):
        rag = await rag_factory(community_levels=2, community_min_size=3)
        rag.reporter = _fake_reporter()
        await _two_cluster_graph(rag)
        await rag.build_communities()
        from post_graph_rag.models import QueryParam
        d1 = await rag.query_data("themes?", param=QueryParam(mode="global",
                                                              community_level=1))
        levels = {c.get("level", 0) for c in d1["data"]["communities"]}
        got0 = await rag.query_data("themes?", param=QueryParam(mode="global",
                                                                community_level=0))
        levels0 = {c.get("level", 0) for c in got0["data"]["communities"]}
        # in-search filtering: a level-restricted query returns that level,
        # never an empty remainder, even when the other level dominates
        assert levels == {1}, d1["data"]["communities"]
        assert levels0 == {0}
