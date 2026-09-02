"""Context assembly must degrade gracefully, never to nothing.

A passage larger than the budget used to break the selection loop before
anything was chosen, so one long document produced an empty context and the
model answered from nothing while holding the text it had retrieved.
"""
from post_graph_rag.engine import _MIN_CLIP_TOKENS, _share, _truncate_by_tokens
from post_graph_rag.models import QueryParam


class TestTruncation:
    def test_oversized_first_passage_is_clipped_not_dropped(self):
        # the regression: 6 sessions x 14k chars against a 2000-token budget
        out = _truncate_by_tokens(["x" * 14057 for _ in range(6)], 2000)
        assert out, "one long passage must not yield an empty context"
        assert len(out) <= 2000 * 4
        assert out.endswith("[truncated]")

    def test_fitting_passages_are_unchanged(self):
        items = ["y" * 2000 for _ in range(3)]
        out = _truncate_by_tokens(items, 2000)
        assert out == "\n".join(items)          # no marker, nothing clipped

    def test_budget_is_respected(self):
        out = _truncate_by_tokens(["z" * 100000], 500)
        assert len(out) <= 500 * 4

    def test_partial_fill_then_clip(self):
        items = ["a" * 4000, "b" * 40000]       # first fits, second overflows
        out = _truncate_by_tokens(items, 2000)
        assert out.startswith("a" * 100)
        assert "b" in out and out.endswith("[truncated]")

    def test_tiny_remainder_is_dropped_not_emitted(self):
        # first passage consumes all but a sliver; a sliver carries no content
        items = ["a" * (2000 * 4), "b" * 40000]
        out = _truncate_by_tokens(items, 2000 + _MIN_CLIP_TOKENS - 1)
        assert "b" not in out

    def test_empty_input(self):
        assert _truncate_by_tokens([], 100) == ""


class TestUnlimitedByDefault:
    """No budget is the default: everything retrieved reaches the model."""

    def test_none_sends_everything(self):
        items = ["x" * 100000 for _ in range(5)]
        out = _truncate_by_tokens(items, None)
        assert out == "\n".join(items)
        assert "[truncated]" not in out

    def test_query_param_defaults_are_unlimited(self):
        p = QueryParam()
        assert p.max_total_tokens is None
        assert p.max_entity_tokens is None
        assert p.max_relation_tokens is None

    def test_share_passes_none_through(self):
        assert _share(None, 2) is None
        assert _share(4000, 2) == 2000

    def test_explicit_budget_still_caps(self):
        # opting back in for cost or a small window must still work
        out = _truncate_by_tokens(["z" * 100000], 500)
        assert len(out) <= 500 * 4
