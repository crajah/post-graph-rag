"""MMR, node-distance reranking and contradiction detection.

The three features taken from Graphiti. Each is tested for the property that
made it worth adding, and for the failure it must not have: MMR must not
promote the irrelevant, distance must not discard what it cannot reach, and
contradiction detection must never retract on a bad answer from the model.
"""
import pytest

from post_graph_rag.engine import GraphRAG
from post_graph_rag.extractor import GraphExtractor


def triple(src, pred, tgt, description="", **extra):
    return {"src_id": src, "relation_type": pred, "tgt_id": tgt,
            "description": description, **extra}


# ------------------------------------------------------------------ MMR

def test_mmr_demotes_a_restatement_of_a_chosen_fact():
    """The case MMR exists for: two channels returning the same fact reworded.

    `_dedupe_triples` removes exact repeats, so the survivor is the near-repeat
    with different wording — and at top_k of a handful it costs a slot that a
    different fact would have used.
    """
    candidates = [
        triple("Alice", "uses", "Disney+", "Alice started a Disney+ subscription"),
        triple("Alice", "subscribed_to", "Disney+", "Alice subscribed to Disney+ streaming"),
        triple("Alice", "lives_in", "Berlin", "Alice moved to Berlin"),
    ]
    out = GraphRAG._apply_mmr(candidates, lambda_=0.5)
    assert out[0] == candidates[0]
    # Berlin overtakes the Disney+ restatement despite ranking below it.
    assert out[1]["tgt_id"] == "Berlin"


def test_mmr_leaves_order_alone_when_nothing_is_redundant():
    candidates = [
        triple("Alice", "lives_in", "Berlin"),
        triple("Bob", "works_at", "Acme"),
        triple("Carol", "plays", "cello"),
    ]
    assert GraphRAG._apply_mmr(candidates, lambda_=0.5) == candidates


def test_mmr_at_lambda_one_is_a_no_op():
    """Pure relevance. The knob has to be able to turn the feature off."""
    candidates = [
        triple("Alice", "uses", "Disney+", "Alice started a Disney+ subscription"),
        triple("Alice", "subscribed_to", "Disney+", "Alice subscribed to Disney+ streaming"),
    ]
    assert GraphRAG._apply_mmr(candidates, lambda_=1.0) == candidates


def test_mmr_never_drops_a_candidate():
    """Reordering, not filtering — truncation happens later against a budget."""
    candidates = [triple(f"E{i}", "rel", "X", "same words repeated") for i in range(6)]
    assert len(GraphRAG._apply_mmr(candidates, lambda_=0.3)) == 6


def test_mmr_handles_an_empty_list():
    assert GraphRAG._apply_mmr([], lambda_=0.5) == []


def test_redundancy_is_symmetric_and_bounded():
    a = triple("Alice", "lives_in", "Berlin", "Alice moved to Berlin")
    b = triple("Alice", "lives_in", "Berlin", "Alice moved to Berlin")
    c = triple("Zed", "plays", "cello", "unrelated entirely")
    assert GraphRAG._redundancy(a, b) == pytest.approx(1.0)
    assert GraphRAG._redundancy(a, c) == GraphRAG._redundancy(c, a)
    assert 0.0 <= GraphRAG._redundancy(a, c) < 0.2


# -------------------------------------------------- node-distance reranking

def test_distance_rerank_demotes_a_far_relation_the_embedding_channel_found():
    """The relation-embedding channel reports hops=1 for everything it finds.

    It never walked, so it has no distance to report. Left alone, a relation
    three hops from anything the question mentioned outranks one sitting on it.
    """
    near = triple("Alice", "lives_in", "Berlin", src_key="a", tgt_key="b")
    far = triple("Xavier", "likes", "opera", src_key="x", tgt_key="y")
    out = GraphRAG._rerank_by_node_distance([far, near], {"a": 0, "b": 1, "x": 3, "y": 3})
    assert out == [near, far]


def test_distance_rerank_keeps_unreached_relations():
    """Not having been reached is why the other channels exist.

    Dropping them would undo the whole point of searching relations directly.
    """
    reached = triple("Alice", "lives_in", "Berlin", src_key="a", tgt_key="b")
    unreached = triple("Nemo", "found_by", "no walk", src_key="zz", tgt_key="yy")
    out = GraphRAG._rerank_by_node_distance([reached, unreached], {"a": 0, "b": 1})
    assert len(out) == 2 and unreached in out


def test_distance_rerank_is_stable_within_a_distance():
    first = triple("A", "r", "B", src_key="a", tgt_key="b")
    second = triple("C", "r", "D", src_key="a", tgt_key="d")
    out = GraphRAG._rerank_by_node_distance([first, second], {"a": 0, "b": 1, "d": 1})
    assert out == [first, second]


def test_distance_rerank_without_distances_changes_nothing():
    """A query that matched no entity has no focal point to measure from."""
    candidates = [triple("A", "r", "B"), triple("C", "r", "D")]
    assert GraphRAG._rerank_by_node_distance(candidates, {}) == candidates


# ------------------------------------------------ contradiction detection

class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def chat_completion(self, messages, **kwargs):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


CANDIDATES = [
    {"id": "10", "relation_type": "lives_in", "from_id": "1", "to_id": "2",
     "description": "Alice lives in Paris"},
    {"id": "11", "relation_type": "owns", "from_id": "1", "to_id": "3",
     "description": "Alice owns a bicycle"},
]


@pytest.mark.asyncio
async def test_contradiction_returns_the_ids_the_model_named():
    ex = GraphExtractor(FakeLLM('{"contradicted_ids": ["10"]}'))
    assert await ex.detect_contradictions("Alice lives in Berlin", CANDIDATES) == ["10"]


@pytest.mark.asyncio
async def test_contradiction_ignores_ids_that_were_never_offered():
    """A hallucinated ID would retract a relation the model never saw."""
    ex = GraphExtractor(FakeLLM('{"contradicted_ids": ["10", "999"]}'))
    assert await ex.detect_contradictions("Alice lives in Berlin", CANDIDATES) == ["10"]


@pytest.mark.asyncio
async def test_contradiction_retracts_nothing_when_the_model_fails():
    """Deliberately not fail-closed.

    Everywhere else in this library a degraded LLM raises rather than silently
    producing less. Here the damage runs the other way: acting on a bad answer
    deletes a true fact from every future query, while doing nothing leaves the
    graph exactly as the deterministic pass left it.
    """
    ex = GraphExtractor(FakeLLM(RuntimeError("router down")))
    assert await ex.detect_contradictions("Alice lives in Berlin", CANDIDATES) == []


@pytest.mark.asyncio
async def test_contradiction_does_not_call_the_model_without_candidates():
    llm = FakeLLM('{"contradicted_ids": []}')
    ex = GraphExtractor(llm)
    assert await ex.detect_contradictions("Alice lives in Berlin", []) == []
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_contradiction_tolerates_a_fenced_json_reply():
    ex = GraphExtractor(FakeLLM('```json\n{"contradicted_ids": ["11"]}\n```'))
    assert await ex.detect_contradictions("Alice sold the bicycle", CANDIDATES) == ["11"]


@pytest.mark.asyncio
async def test_contradiction_survives_unparseable_output():
    ex = GraphExtractor(FakeLLM("I think fact 10 is wrong now"))
    assert await ex.detect_contradictions("Alice lives in Berlin", CANDIDATES) == []
