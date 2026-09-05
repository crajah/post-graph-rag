"""Tests for post_graph_rag.models — pure unit tests, no DB or LLM."""

import json

from post_graph_rag.models import (
    DocumentContext,
    DocumentMetadata,
    KeywordResult,
    QueryParam,
    content_hash,
    document_key,
)


class TestDocumentMetadata:
    def test_defaults(self):
        m = DocumentMetadata()
        assert m.source is None
        assert m.category is None
        assert m.page is None
        assert m.extra == {}

    def test_to_dict_omits_none(self):
        m = DocumentMetadata(source="http://x.com", page=3)
        d = m.to_dict()
        assert d["source"] == "http://x.com"
        assert d["page"] == 3
        assert "category" not in d
        assert "collection" not in d

    def test_extra_fields_merged_into_to_dict(self):
        m = DocumentMetadata(source="s", extra={"custom_field": "val"})
        d = m.to_dict()
        assert d["custom_field"] == "val"
        assert d["source"] == "s"

    def test_from_dict_round_trip(self):
        original = DocumentMetadata(
            source="http://example.com", category="reports",
            collection="q3", document="report.pdf", page=5,
            paragraph=2, space="prod",
        )
        d = original.to_dict()
        restored = DocumentMetadata.from_dict(d)
        assert restored.source == original.source
        assert restored.category == original.category
        assert restored.page == original.page

    def test_from_dict_handles_extra_keys(self):
        data = {"source": "s", "author": "Alice", "custom": 42}
        m = DocumentMetadata.from_dict(data)
        assert m.source == "s"
        assert m.extra["author"] == "Alice"
        assert m.extra["custom"] == 42

    def test_from_dict_empty(self):
        m = DocumentMetadata.from_dict({})
        assert m.source is None
        assert m.extra == {}

    def test_from_dict_none(self):
        m = DocumentMetadata.from_dict(None)
        assert m.source is None

    def test_to_dict_is_json_serializable(self):
        m = DocumentMetadata(source="s", page=1, extra={"nested": {"a": [1]}})
        serialized = json.dumps(m.to_dict())
        assert isinstance(serialized, str)

    def test_extra_default_is_independent(self):
        m1 = DocumentMetadata()
        m2 = DocumentMetadata()
        m1.extra["key"] = "val"
        assert "key" not in m2.extra

    def test_identity_fields(self):
        m = DocumentMetadata(doc_key="my_doc", content_hash="abc123")
        d = m.to_dict()
        assert d["doc_key"] == "my_doc"
        assert d["content_hash"] == "abc123"


class TestDocumentContext:
    def test_defaults(self):
        ctx = DocumentContext()
        assert ctx.title is None
        assert ctx.source is None
        assert ctx.summary is None
        assert ctx.known_entities == []

    def test_to_dict(self):
        ctx = DocumentContext(title="T", source="S", summary="Sum",
                              known_entities=["Alice", "Bob"])
        d = ctx.to_dict()
        assert d["title"] == "T"
        assert d["known_entities"] == ["Alice", "Bob"]

    def test_known_entities_independent(self):
        c1 = DocumentContext()
        c2 = DocumentContext()
        c1.known_entities.append("X")
        assert "X" not in c2.known_entities


class TestKeywordResult:
    def test_defaults(self):
        k = KeywordResult()
        assert k.high_level_keywords == []
        assert k.low_level_keywords == []

    def test_to_dict(self):
        k = KeywordResult(
            high_level_keywords=["theme"],
            low_level_keywords=["entity"]
        )
        d = k.to_dict()
        assert d["high_level"] == ["theme"]
        assert d["low_level"] == ["entity"]

    def test_lists_are_independent(self):
        k1 = KeywordResult()
        k2 = KeywordResult()
        k1.high_level_keywords.append("x")
        assert "x" not in k2.high_level_keywords


class TestQueryParam:
    def test_defaults(self):
        p = QueryParam()
        assert p.mode == "mix"
        assert p.top_k == 5
        assert p.stream is False
        assert p.only_need_context is False
        assert p.space is None
        assert p.as_of is None
        assert p.conversation_history == []

    def test_override_fields(self):
        p = QueryParam(mode="global", top_k=10, stream=True, as_of="1625")
        assert p.mode == "global"
        assert p.top_k == 10
        assert p.stream is True
        assert p.as_of == "1625"

    def test_conversation_history_independent(self):
        p1 = QueryParam()
        p2 = QueryParam()
        p1.conversation_history.append({"role": "user", "content": "hi"})
        assert len(p2.conversation_history) == 0


class TestDocumentKey:
    def test_uses_both_parts(self):
        # Source alone used to win, which let a constant source collapse an
        # entire corpus onto one key — and a matching key means re-index.
        assert document_key("http://x.com", "doc.pdf") == "http://x.com::doc.pdf"

    def test_distinct_documents_under_one_source_stay_distinct(self):
        assert document_key("corpus", "a") != document_key("corpus", "b")

    def test_falls_back_to_document(self):
        assert document_key(None, "doc.pdf") == "doc.pdf"

    def test_strips_whitespace(self):
        assert document_key("  http://x.com  ", None) == "http://x.com"

    def test_defaults_to_unkeyed(self):
        assert document_key(None, None) == "unkeyed"

    def test_empty_strings_give_unkeyed(self):
        assert document_key("", "") == "unkeyed"

    def test_whitespace_only_gives_unkeyed(self):
        assert document_key("   ", "   ") == "unkeyed"


class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_for_different_text(self):
        assert content_hash("hello") != content_hash("world")

    def test_is_hex_string(self):
        h = content_hash("test")
        assert len(h) == 32
        int(h, 16)  # should not raise

    def test_empty_text(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) == 32
