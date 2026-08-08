"""Tests for community detection, reports, and community-backed global retrieval."""
import json

import pytest

from post_graph_rag import QueryParam, RAGConfig
from post_graph_rag.communities import (
    default_detector, group_by_community, label_propagation,
)
from post_graph_rag.errors import ExtractionError
from post_graph_rag.extractor import Entity, ExtractionResult, Triple
from post_graph_rag.reporting import (
    CommunityReport, CommunityReporter, Finding, render_community, report_to_text,
)

from conftest import FakeLLM

# Two dense clusters joined by nothing. Any reasonable detector must separate them.
TWO_CLUSTERS_NODES = ["a1", "a2", "a3", "b1", "b2", "b3"]
TWO_CLUSTERS_EDGES = [
    ("a1", "a2", 1.0), ("a2", "a3", 1.0), ("a3", "a1", 1.0),
    ("b1", "b2", 1.0), ("b2", "b3", 1.0), ("b3", "b1", 1.0),
]


# ---------------------------------------------------------------- detection

def test_label_propagation_separates_disconnected_clusters():
    assignment = label_propagation(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES)
    groups = group_by_community(assignment, min_size=1)
    assert len(groups) == 2
    members = sorted(sorted(v) for v in groups.values())
    assert members == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]


def test_label_propagation_is_deterministic():
    """A randomised partition would give a different graph on every index run."""
    first = label_propagation(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES)
    for _ in range(5):
        assert label_propagation(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES) == first


def test_detector_handles_isolated_and_self_looping_nodes():
    nodes = ["a", "b", "lonely"]
    edges = [("a", "b", 1.0), ("a", "a", 5.0), ("a", "ghost", 1.0)]
    assignment = label_propagation(nodes, edges)
    assert set(assignment) == set(nodes)
    assert "ghost" not in assignment


def test_default_detector_returns_full_assignment():
    assignment = default_detector(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES, resolution=1.0)
    assert set(assignment) == set(TWO_CLUSTERS_NODES)


def test_default_detector_falls_back_without_leiden(monkeypatch):
    """Leiden is optional; its absence must not break community building."""
    import post_graph_rag.communities as mod

    def _no_leiden(*a, **kw):
        raise ImportError("no igraph")

    monkeypatch.setattr(mod, "leiden", _no_leiden)
    assignment = mod.default_detector(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES)
    assert len(group_by_community(assignment, min_size=1)) == 2


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("leidenalg") is None,
    reason="leidenalg not installed",
)
def test_leiden_separates_disconnected_clusters():
    from post_graph_rag.communities import leiden
    groups = group_by_community(leiden(TWO_CLUSTERS_NODES, TWO_CLUSTERS_EDGES), min_size=1)
    assert len(groups) == 2


def test_group_by_community_drops_small_groups():
    assignment = {"a": 0, "b": 0, "c": 0, "d": 1}
    assert sorted(group_by_community(assignment, min_size=3)) == [0]
    assert sorted(group_by_community(assignment, min_size=1)) == [0, 1]


def test_empty_graph_is_safe():
    assert label_propagation([], []) == {}
    assert group_by_community({}, min_size=2) == {}


# ------------------------------------------------------------------ reports

def _reporter(report=None, fail=False):
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    return CommunityReporter(FakeLLM(config, extraction=report, fail=fail))


REPORT = CommunityReport(
    title="Babbage's calculating engines",
    summary="Charles Babbage designed mechanical computers.",
    findings=[Finding(summary="Analytical Engine", explanation="A general-purpose design.")],
    rating=8.5,
)


@pytest.mark.asyncio
async def test_reporter_produces_report():
    out = await _reporter(REPORT).summarise(
        [{"name": "Charles Babbage", "type": "Person", "description": "inventor"}],
        [{"src": "Charles Babbage", "tgt": "Analytical Engine", "predicate": "designed",
          "description": "", "weight": 1, "negated": False}],
    )
    assert out.title == "Babbage's calculating engines"
    assert out.rating == 8.5


@pytest.mark.asyncio
async def test_reporter_raises_when_llm_fails():
    """A fabricated report would be indistinguishable from a real one."""
    with pytest.raises(Exception):
        await _reporter(fail=True).summarise([{"name": "X"}], [])


@pytest.mark.asyncio
async def test_reporter_rejects_empty_summary():
    empty = CommunityReport(title="t", summary="   ", findings=[], rating=1.0)
    with pytest.raises(ExtractionError):
        await _reporter(empty).summarise([{"name": "X"}], [])


def test_render_community_marks_negated_relations():
    """A denied relation must not read as an assertion in the prompt."""
    text = render_community(
        [{"name": "Ada Lovelace", "type": "Person", "description": "d"}],
        [{"src": "Ada Lovelace", "tgt": "Lord Byron", "predicate": "had_relationship_with",
          "description": "estranged", "weight": 1, "negated": True}],
    )
    assert "[NOT had_relationship_with]" in text


def test_report_to_text_includes_findings():
    text = report_to_text(REPORT)
    assert "Babbage's calculating engines" in text
    assert "Analytical Engine" in text
    assert "A general-purpose design." in text


# ---------------------------------------------------------- DB-backed build

CLUSTER_A = ExtractionResult(
    entities=[
        Entity(name="Charles Babbage", type="Person", description="inventor"),
        Entity(name="Analytical Engine", type="Technology", description="mechanical computer"),
        Entity(name="Difference Engine", type="Technology", description="calculating machine"),
    ],
    triples=[
        Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine"),
        Triple(subject="Charles Babbage", predicate="designed", object="Difference Engine"),
        Triple(subject="Analytical Engine", predicate="succeeded", object="Difference Engine"),
    ],
)


@pytest.mark.asyncio
async def test_build_communities_creates_reports(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")

    res = await rag.build_communities()
    assert res["communities"] >= 1
    assert await rag.store.count_communities() == res["communities"]

    rows = await rag.store.client._fetch(
        f'SELECT payload FROM "{rag.config.realm}"."communities"')
    payload = rows[0]["payload"]
    payload = payload if isinstance(payload, dict) else json.loads(payload)
    assert payload["title"] == "Babbage's calculating engines"
    assert payload["size"] == 3
    # Membership edges connect the report to its entities.
    members = await rag.store.client._fetch(
        f'SELECT count(*) AS n FROM "{rag.config.realm}"."community_members"')
    assert members[0]["n"] >= 3


@pytest.mark.asyncio
async def test_build_communities_is_idempotent(rag_factory):
    """Communities are derived data: rebuilding replaces rather than accumulates."""
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")

    first = await rag.build_communities()
    second = await rag.build_communities()
    assert first["communities"] == second["communities"]
    assert await rag.store.count_communities() == second["communities"]


@pytest.mark.asyncio
async def test_build_communities_on_empty_graph(rag_factory):
    rag = await rag_factory()
    res = await rag.build_communities()
    assert res["communities"] == 0


@pytest.mark.asyncio
async def test_min_size_skips_small_communities(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A, community_min_size=99)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")
    res = await rag.build_communities()
    assert res["communities"] == 0


@pytest.mark.asyncio
async def test_total_report_failure_raises(rag_factory):
    """Skipping some communities is recovery; skipping every one is an outage.

    Observed live: a 12-community build reported 'summarised 0, skipped 12' as a
    successful result after 38 minutes of retries against a failing endpoint.
    """
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(fail=True)
    await rag.index_document("Babbage designed engines.")
    with pytest.raises(Exception):
        await rag.build_communities()


CLUSTER_B = ExtractionResult(
    entities=[
        Entity(name="Marie Curie", type="Person", description="physicist"),
        Entity(name="Radium", type="Concept", description="element"),
        Entity(name="Polonium", type="Concept", description="element"),
    ],
    triples=[
        Triple(subject="Marie Curie", predicate="discovered", object="Radium"),
        Triple(subject="Marie Curie", predicate="discovered", object="Polonium"),
        Triple(subject="Radium", predicate="related_element", object="Polonium"),
    ],
)


@pytest.mark.asyncio
async def test_unusable_report_skips_rather_than_aborting(rag_factory):
    """One bad report must not abandon the build, as long as others succeed.

    Needs two disconnected clusters: with a single community, a failed report
    means nothing was built at all, which correctly raises instead.
    """
    rag = await rag_factory(extraction=CLUSTER_A)
    await rag.index_document("Babbage designed engines.")
    rag.llm._extraction = CLUSTER_B
    await rag.index_document("Curie discovered radium.")

    calls = {"n": 0}
    good = _reporter(REPORT)

    class Flaky:
        async def summarise(self, entities, relations):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ExtractionError("first one is unusable")
            return await good.summarise(entities, relations)

    rag.reporter = Flaky()
    res = await rag.build_communities()
    assert res["skipped"] == 1
    assert res["communities"] >= 1, "a survivable failure aborted the whole build"


def test_colliding_titles_are_disambiguated():
    """Reports are generated independently, so titles can collide; retrieval
    would then show the same label for two different subgraphs."""
    from post_graph_rag.engine import GraphRAG

    used = {}
    members = [{"name": "Analytical Engine"}, {"name": "Henry Prevost Babbage"}]
    first = GraphRAG._disambiguate_title("The Analytical Engine Community", members, used)
    second = GraphRAG._disambiguate_title("The Analytical Engine Community", members, used)
    assert first == "The Analytical Engine Community"
    assert second == "The Analytical Engine Community (Henry Prevost Babbage)"
    assert first != second


def test_disambiguation_falls_back_to_a_counter():
    from post_graph_rag.engine import GraphRAG

    used = {}
    GraphRAG._disambiguate_title("Engines", [{"name": "Engines"}], used)
    assert GraphRAG._disambiguate_title("Engines", [{"name": "Engines"}], used) == "Engines #2"


@pytest.mark.asyncio
async def test_build_disambiguates_duplicate_titles(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A, community_min_size=1)
    rag.reporter = _reporter(REPORT)   # always returns the same title
    await rag.index_document("Babbage designed engines.")
    res = await rag.build_communities()

    rows = await rag.store.client._fetch(
        f'SELECT payload FROM "{rag.config.realm}"."communities"')
    titles = []
    for r in rows:
        p = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
        titles.append(p["title"])
    assert len(titles) == len(set(titles)), titles
    assert res["communities"] == len(titles)


# ------------------------------------------------------------ global search

@pytest.mark.asyncio
async def test_global_mode_returns_community_reports(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")
    await rag.build_communities()

    res = await rag.query_data("What are the main themes?", param=QueryParam(mode="global", top_k=3))
    assert res["data"]["communities"]
    assert res["data"]["communities"][0]["title"] == "Babbage's calculating engines"
    assert res["metadata"]["processing_info"]["communities_found"] >= 1


@pytest.mark.asyncio
async def test_global_mode_without_communities_still_works(rag_factory):
    """Degrades to relation ranking rather than failing when none were built."""
    rag = await rag_factory(extraction=CLUSTER_A)
    await rag.index_document("Babbage designed engines.")
    res = await rag.query_data("themes?", param=QueryParam(mode="global", top_k=3))
    assert res["data"]["communities"] == []
    assert res["status"] == "success"


@pytest.mark.asyncio
async def test_local_mode_does_not_fetch_communities(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")
    await rag.build_communities()
    res = await rag.query_data("Babbage", param=QueryParam(mode="local", top_k=3))
    assert res["data"]["communities"] == []


@pytest.mark.asyncio
async def test_community_summaries_reach_synthesis(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A, answer="synthesised")
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.")
    await rag.build_communities()

    res = await rag.query("What are the main themes?", param=QueryParam(mode="global", top_k=3))
    prompt = rag.llm.chat_calls[-1][0][-1]["content"]
    assert "Knowledge Base Themes" in prompt
    assert "Babbage's calculating engines" in prompt
    assert res["retrieved_communities"] == ["Babbage's calculating engines"]


@pytest.mark.asyncio
async def test_communities_are_space_scoped(rag_factory):
    rag = await rag_factory(extraction=CLUSTER_A)
    rag.reporter = _reporter(REPORT)
    await rag.index_document("Babbage designed engines.", space="production")
    await rag.build_communities(space="production")

    prod = await rag.query_data("themes", param=QueryParam(mode="global", top_k=3, space="production"))
    sand = await rag.query_data("themes", param=QueryParam(mode="global", top_k=3, space="sandbox"))
    assert prod["data"]["communities"]
    assert sand["data"]["communities"] == []
