"""Tests for post_graph_rag.reporting — report rendering and parsing."""

import json


from post_graph_rag.reporting import (
    CommunityReport,
    CommunityReporter,
    Finding,
    render_community,
    report_to_text,
    REPORT_SYSTEM_PROMPT,
)


class TestFinding:
    def test_fields(self):
        f = Finding(summary="Key insight", explanation="Because reasons")
        assert f.summary == "Key insight"
        assert f.explanation == "Because reasons"


class TestCommunityReport:
    def test_defaults(self):
        r = CommunityReport(title="T", summary="S")
        assert r.title == "T"
        assert r.summary == "S"
        assert r.findings == []
        assert r.rating == 5.0

    def test_with_findings(self):
        f = Finding(summary="F1", explanation="E1")
        r = CommunityReport(title="T", summary="S", findings=[f], rating=8.0)
        assert len(r.findings) == 1
        assert r.rating == 8.0


class TestRenderCommunity:
    def test_basic_render(self):
        entities = [{"name": "Alice", "type": "Person", "description": "A person"}]
        relations = [
            {"src": "Alice", "tgt": "Bob", "predicate": "knows",
             "weight": 1, "negated": False, "description": "They know each other"}
        ]
        text = render_community(entities, relations)
        assert "Alice" in text
        assert "knows" in text
        assert "Bob" in text

    def test_negated_relation_marked(self):
        entities = [{"name": "A", "type": "X", "description": ""}]
        relations = [
            {"src": "A", "tgt": "B", "predicate": "knows",
             "weight": 1, "negated": True, "description": ""}
        ]
        text = render_community(entities, relations)
        assert "NOT " in text

    def test_truncates_entities(self):
        entities = [{"name": f"E{i}", "type": "T", "description": ""} for i in range(100)]
        text = render_community(entities, [], max_entities=5)
        assert "E4" in text
        assert "E5" not in text

    def test_truncates_relations(self):
        relations = [
            {"src": f"S{i}", "tgt": f"T{i}", "predicate": "r",
             "weight": 1, "negated": False, "description": ""}
            for i in range(100)
        ]
        text = render_community([], relations, max_relations=3)
        assert "S2" in text
        assert "S3" not in text

    def test_missing_fields_handled(self):
        entities = [{"name": None, "type": None, "description": None}]
        relations = [{"src": None, "tgt": None, "predicate": None,
                       "weight": None, "negated": False, "description": None}]
        text = render_community(entities, relations)
        assert "Entities:" in text
        assert "Relations:" in text


class TestReportToText:
    def test_includes_title_and_summary(self):
        r = CommunityReport(title="Olympians", summary="Greek gods and their domains")
        text = report_to_text(r)
        assert "Olympians" in text
        assert "Greek gods" in text

    def test_includes_findings(self):
        f = Finding(summary="Zeus rules", explanation="King of the gods")
        r = CommunityReport(title="T", summary="S", findings=[f])
        text = report_to_text(r)
        assert "Zeus rules" in text
        assert "King of the gods" in text

    def test_no_findings(self):
        r = CommunityReport(title="T", summary="S")
        text = report_to_text(r)
        lines = [line for line in text.split("\n") if line.strip()]
        assert len(lines) == 2


class TestCommunityReporterParse:
    def test_parse_valid_json(self):
        data = json.dumps({
            "title": "Test", "summary": "A summary",
            "findings": [{"summary": "F1", "explanation": "E1"}],
            "rating": 7.5
        })
        result = CommunityReporter._parse(data)
        assert isinstance(result, CommunityReport)
        assert result.title == "Test"
        assert result.rating == 7.5

    def test_parse_json_with_fences(self):
        data = "```json\n" + json.dumps({"title": "T", "summary": "S"}) + "\n```"
        result = CommunityReporter._parse(data)
        assert isinstance(result, CommunityReport)
        assert result.title == "T"

    def test_parse_empty_returns_none(self):
        assert CommunityReporter._parse("") is None
        assert CommunityReporter._parse("   ") is None
        assert CommunityReporter._parse(None) is None

    def test_parse_invalid_json_returns_none(self):
        assert CommunityReporter._parse("not json at all") is None

    def test_parse_incomplete_json_returns_none(self):
        assert CommunityReporter._parse('{"title": "T"') is None


class TestReportSystemPrompt:
    def test_prompt_mentions_rules(self):
        assert "RULES" in REPORT_SYSTEM_PROMPT
        assert "NOT" in REPORT_SYSTEM_PROMPT

    def test_prompt_is_nonempty_string(self):
        assert isinstance(REPORT_SYSTEM_PROMPT, str)
        assert len(REPORT_SYSTEM_PROMPT) > 100
