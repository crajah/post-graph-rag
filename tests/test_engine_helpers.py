"""Tests for engine-level helper functions — pure unit tests."""


from post_graph_rag.engine import (
    _valid_at,
    _truncate_by_tokens,
    _accepts_resolution,
    GraphRAG,
    RETRIEVAL_MODES,
)


class TestValidAt:
    def test_undated_always_valid(self):
        assert _valid_at(None, None, "1625") is True

    def test_within_range(self):
        assert _valid_at("1620", "1630", "1625") is True

    def test_before_range(self):
        assert _valid_at("1630", "1640", "1625") is False

    def test_after_range(self):
        assert _valid_at("1620", "1624", "1625") is False

    def test_only_from(self):
        assert _valid_at("1620", None, "1625") is True
        assert _valid_at("1630", None, "1625") is False

    def test_only_to(self):
        assert _valid_at(None, "1630", "1625") is True
        assert _valid_at(None, "1620", "1625") is False

    def test_exact_boundary(self):
        assert _valid_at("1625", "1625", "1625") is True

    def test_partial_dates(self):
        assert _valid_at("1625-06", None, "1625-06-15") is True


class TestTruncateByTokens:
    def test_empty(self):
        assert _truncate_by_tokens([], 100) == ""

    def test_fits(self):
        items = ["short", "text"]
        result = _truncate_by_tokens(items, 100)
        assert "short" in result
        assert "text" in result

    def test_truncates(self):
        items = ["a" * 100, "b" * 100, "c" * 100]
        result = _truncate_by_tokens(items, 30)
        assert "a" * 100 in result
        assert "c" * 100 not in result

    def test_single_large_item(self):
        items = ["x" * 1000]
        result = _truncate_by_tokens(items, 1)
        assert result == ""


class TestAcceptsResolution:
    def test_with_resolution_param(self):
        def detector(nodes, edges, resolution=1.0):
            pass
        assert _accepts_resolution(detector) is True

    def test_without_resolution_param(self):
        def detector(nodes, edges):
            pass
        assert _accepts_resolution(detector) is False

    def test_lambda(self):
        assert _accepts_resolution(lambda nodes, edges: {}) is False


class TestDedupeTriples:
    def test_removes_duplicates(self):
        triples = [
            {"src_id": "A", "relation_type": "knows", "tgt_id": "B", "weight": 1},
            {"src_id": "A", "relation_type": "knows", "tgt_id": "B", "weight": 2},
        ]
        result = GraphRAG._dedupe_triples(triples)
        assert len(result) == 1
        assert result[0]["weight"] == 1  # first occurrence kept

    def test_case_insensitive(self):
        triples = [
            {"src_id": "alice", "relation_type": "Knows", "tgt_id": "BOB"},
            {"src_id": "Alice", "relation_type": "knows", "tgt_id": "Bob"},
        ]
        result = GraphRAG._dedupe_triples(triples)
        assert len(result) == 1

    def test_different_triples_kept(self):
        triples = [
            {"src_id": "A", "relation_type": "knows", "tgt_id": "B"},
            {"src_id": "A", "relation_type": "works_at", "tgt_id": "B"},
        ]
        result = GraphRAG._dedupe_triples(triples)
        assert len(result) == 2

    def test_empty(self):
        assert GraphRAG._dedupe_triples([]) == []


class TestMergeByQuota:
    def test_empty_seeded(self):
        traversed = [{"src_id": "A"}]
        result = GraphRAG._merge_by_quota(traversed, [], 0.5)
        assert result == traversed

    def test_empty_traversed(self):
        seeded = [{"src_id": "B"}]
        result = GraphRAG._merge_by_quota([], seeded, 0.5)
        assert result == seeded

    def test_interleaves(self):
        traversed = [{"src_id": f"T{i}"} for i in range(4)]
        seeded = [{"src_id": f"S{i}"} for i in range(4)]
        result = GraphRAG._merge_by_quota(traversed, seeded, 0.5)
        assert len(result) == 8
        # Both channels represented
        t_count = sum(1 for r in result if r["src_id"].startswith("T"))
        s_count = sum(1 for r in result if r["src_id"].startswith("S"))
        assert t_count == 4
        assert s_count == 4

    def test_quota_zero_all_traversed_first(self):
        traversed = [{"src_id": f"T{i}"} for i in range(3)]
        seeded = [{"src_id": f"S{i}"} for i in range(3)]
        result = GraphRAG._merge_by_quota(traversed, seeded, 0.0)
        # Traversed should come first when quota is 0
        for i in range(3):
            assert result[i]["src_id"].startswith("T")

    def test_quota_one_all_seeded_first(self):
        traversed = [{"src_id": f"T{i}"} for i in range(3)]
        seeded = [{"src_id": f"S{i}"} for i in range(3)]
        result = GraphRAG._merge_by_quota(traversed, seeded, 1.0)
        for i in range(3):
            assert result[i]["src_id"].startswith("S")


class TestRankByKeywords:
    def test_ranks_by_overlap(self):
        triples = [
            {"src_id": "Unrelated", "tgt_id": "Nope", "relation_type": "r", "description": "", "weight": 1},
            {"src_id": "Zeus", "tgt_id": "Olympus", "relation_type": "rules", "description": "king of gods", "weight": 1},
        ]
        result = GraphRAG._rank_by_keywords(triples, ["Zeus", "gods"], 10)
        assert result[0]["src_id"] == "Zeus"

    def test_respects_limit(self):
        triples = [{"src_id": f"E{i}", "tgt_id": "T", "relation_type": "r", "description": "", "weight": 1} for i in range(10)]
        result = GraphRAG._rank_by_keywords(triples, ["E1"], 3)
        assert len(result) == 3

    def test_empty_keywords(self):
        triples = [{"src_id": "A", "tgt_id": "B", "relation_type": "r", "description": "", "weight": 1}]
        result = GraphRAG._rank_by_keywords(triples, [], 10)
        assert len(result) == 1

    def test_weight_breaks_ties(self):
        triples = [
            {"src_id": "A", "tgt_id": "B", "relation_type": "r", "description": "", "weight": 1},
            {"src_id": "A", "tgt_id": "B", "relation_type": "r", "description": "", "weight": 5},
        ]
        result = GraphRAG._rank_by_keywords(triples, [], 10)
        assert result[0]["weight"] == 5


class TestDisambiguateTitle:
    def test_first_use_unchanged(self):
        used = {}
        result = GraphRAG._disambiguate_title("Olympians", [], used)
        assert result == "Olympians"
        assert used["Olympians"] == 1

    def test_second_use_qualified(self):
        used = {"Olympians": 1}
        members = [{"name": "Zeus"}, {"name": "Hera"}]
        result = GraphRAG._disambiguate_title("Olympians", members, used)
        assert "Zeus" in result or "Hera" in result

    def test_member_already_in_title_skipped(self):
        used = {"Zeus and the Olympians": 1}
        members = [{"name": "Zeus"}, {"name": "Hera"}]
        result = GraphRAG._disambiguate_title("Zeus and the Olympians", members, used)
        # Zeus is in the title, so should use Hera or fall back to counter
        assert "Hera" in result or "#" in result

    def test_no_members_falls_back_to_counter(self):
        used = {"Title": 1}
        result = GraphRAG._disambiguate_title("Title", [], used)
        assert result == "Title #2"

    def test_empty_title(self):
        used = {}
        result = GraphRAG._disambiguate_title("", [], used)
        assert result == "Community"


class TestRetrievalModes:
    def test_known_modes(self):
        assert "mix" in RETRIEVAL_MODES
        assert "local" in RETRIEVAL_MODES
        assert "global" in RETRIEVAL_MODES
        assert "hybrid" in RETRIEVAL_MODES
        assert "naive" in RETRIEVAL_MODES
        assert "bypass" in RETRIEVAL_MODES
