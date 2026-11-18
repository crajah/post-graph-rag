"""Live-PostgreSQL tests for bi-temporal recall_as_of, trace_history, changes_since, rules, and realm guards."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest
from post_graph_rag.errors import SchemaError


async def test_belief_time_restatement_audit_returns_prior_belief_before_supersession(
    mcp_harness_factory: Callable[..., Any],
) -> None:
    harness = mcp_harness_factory(preset="general_memory")

    initial_cfo = await harness.call(
        "remember",
        {
            "text": "Calder appointed Voss as Chief Financial Officer.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Voss",
                }
            ],
        },
    )
    initial_rev = await harness.call(
        "remember",
        {
            "text": "Calder_2019 reported revenue of $412 million in its initial filing.",
            "triples": [
                {
                    "subject": "Calder_2019",
                    "predicate": "reported_revenue",
                    "object": "$412 million",
                    "functional": True,
                }
            ],
        },
    )
    watermark_before_restatement = initial_rev["recorded_at_watermark"]
    assert watermark_before_restatement >= initial_cfo["recorded_at_watermark"]

    await asyncio.sleep(0.03)

    await harness.call(
        "remember",
        {
            "text": "Calder appointed Nair to succeed Voss as Chief Financial Officer.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Nair",
                }
            ],
        },
    )
    await harness.call(
        "remember",
        {
            "text": "Calder_2019 restated revenue to $389 million following audit review.",
            "triples": [
                {
                    "subject": "Calder_2019",
                    "predicate": "reported_revenue",
                    "object": "$389 million",
                    "functional": True,
                }
            ],
        },
    )

    now_cfo = await harness.call(
        "recall",
        {
            "question": "Who is the CFO of Calder?",
            "entity_keywords": ["Calder"],
        },
    )
    assert [
        harness.object(r)
        for r in now_cfo["relationships"]
        if harness.subject(r) == "Calder"
    ] == ["Nair"]

    now_rev = await harness.call(
        "recall",
        {
            "question": "What revenue did Calder_2019 report?",
            "entity_keywords": ["Calder_2019"],
        },
    )
    assert [
        harness.object(r)
        for r in now_rev["relationships"]
        if harness.subject(r) == "Calder_2019"
    ] == ["$389 million"]

    believed_cfo = await harness.call(
        "recall_as_of",
        {
            "question": "Who did we believe was the CFO of Calder?",
            "as_believed_at": watermark_before_restatement,
            "entity_keywords": ["Calder"],
        },
    )
    assert believed_cfo["temporal_axes_applied"] == {
        "as_of": None,
        "as_believed_at": watermark_before_restatement,
    }
    believed_cfo_objs = [
        harness.object(r)
        for r in believed_cfo["relationships"]
        if harness.subject(r) == "Calder"
    ]
    assert believed_cfo_objs == ["Voss"]

    believed_rev = await harness.call(
        "recall_as_of",
        {
            "question": "What revenue was believed reported by Calder_2019?",
            "as_believed_at": watermark_before_restatement,
            "entity_keywords": ["Calder_2019"],
        },
    )
    believed_rev_objs = [
        harness.object(r)
        for r in believed_rev["relationships"]
        if harness.subject(r) == "Calder_2019"
    ]
    assert believed_rev_objs == ["$412 million"]

    with pytest.raises(ValueError, match="as_of|as_believed_at"):
        await harness.call(
            "recall_as_of",
            {
                "question": "Who was CFO of Calder?",
                "entity_keywords": ["Calder"],
            },
        )


async def test_valid_time_audit_retains_superseded_interval_covering_target_date(
    mcp_harness_factory: Callable[..., Any],
) -> None:
    harness = mcp_harness_factory(preset="general_memory")

    await harness.call(
        "remember",
        {
            "text": "Voss served as CFO of Calder from 2019-01-01 through 2020-12-31.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Voss",
                    "valid_from": "2019-01-01",
                    "valid_to": "2020-12-31",
                }
            ],
        },
    )
    await harness.call(
        "remember",
        {
            "text": "Chandran served as CFO of Calder from 2021-01-01 through 2022-12-31.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Chandran",
                    "valid_from": "2021-01-01",
                    "valid_to": "2022-12-31",
                }
            ],
        },
    )

    in_2019 = await harness.call(
        "recall_as_of",
        {
            "question": "Who served as CFO of Calder in mid-2019?",
            "as_of": "2019-06-01",
            "entity_keywords": ["Calder"],
        },
    )
    objs_2019 = [
        harness.object(r)
        for r in in_2019["relationships"]
        if harness.subject(r) == "Calder"
    ]
    assert objs_2019 == ["Voss"]

    in_2021 = await harness.call(
        "recall_as_of",
        {
            "question": "Who served as CFO of Calder in mid-2021?",
            "as_of": "2021-06-01",
            "entity_keywords": ["Calder"],
        },
    )
    objs_2021 = [
        harness.object(r)
        for r in in_2021["relationships"]
        if harness.subject(r) == "Calder"
    ]
    assert objs_2021 == ["Chandran"]


async def test_trace_history_outward_subject_and_inward_object_traversal(
    mcp_harness_factory: Callable[..., Any],
) -> None:
    corp_harness = mcp_harness_factory(preset="general_memory")
    await corp_harness.call(
        "remember",
        {
            "text": "Voss served as CFO of Calder in 2019-2020.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Voss",
                    "valid_from": "2019-01-01",
                    "valid_to": "2020-12-31",
                }
            ],
        },
    )
    await corp_harness.call(
        "remember",
        {
            "text": "Chandran became CFO of Calder in 2021.",
            "triples": [
                {
                    "subject": "Calder",
                    "predicate": "has_cfo",
                    "object": "Chandran",
                    "valid_from": "2021-01-01",
                }
            ],
        },
    )
    outward = await corp_harness.call(
        "trace_history",
        {
            "question": "Who has served as CFO of Calder?",
            "entity_keywords": ["Calder"],
        },
    )
    assert [
        corp_harness.object(r)
        for r in outward["superseded_relations"]
        if corp_harness.subject(r) == "Calder"
    ] == ["Voss"]
    assert [
        corp_harness.object(r)
        for r in outward["current_relations"]
        if corp_harness.subject(r) == "Calder"
    ] == ["Chandran"]

    eng_harness = mcp_harness_factory(preset="engineering")
    await eng_harness.call(
        "remember",
        {
            "text": "AuthService initially used Clerk.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "Clerk",
                }
            ],
        },
    )
    await eng_harness.call(
        "remember",
        {
            "text": "AuthService migrated from Clerk to BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "BetterAuth",
                }
            ],
        },
    )
    await eng_harness.call(
        "remember",
        {
            "text": "PartnerPortal still uses Clerk for external SSO.",
            "triples": [
                {
                    "subject": "PartnerPortal",
                    "predicate": "uses_auth",
                    "object": "Clerk",
                }
            ],
        },
    )

    inward = await eng_harness.call(
        "trace_history",
        {
            "question": "Which services use or previously used Clerk?",
            "entity_keywords": ["Clerk"],
        },
    )
    inward_superseded = [
        (eng_harness.subject(r), eng_harness.object(r))
        for r in inward["superseded_relations"]
    ]
    inward_current = [
        (eng_harness.subject(r), eng_harness.object(r))
        for r in inward["current_relations"]
    ]
    assert ("AuthService", "Clerk") in inward_superseded
    assert ("PartnerPortal", "Clerk") in inward_current


async def test_changes_since_delta_feed_and_configure_memory_rules(
    mcp_harness: Any,
) -> None:
    baseline = await mcp_harness.call("changes_since", {"since": None})
    w0 = baseline["watermark"]
    assert w0
    assert baseline["changes"] is None

    await asyncio.sleep(0.02)

    rules_res = await mcp_harness.call(
        "configure_memory_rules",
        {
            "add_functional_predicates": ["uses_cache"],
            "add_exclusive_groups": [["feature_enabled_for", "feature_disabled_for"]],
            "include_diagnostics": True,
        },
    )
    rules_body = rules_res.get("rules", rules_res)
    assert "uses_cache" in set(rules_body["functional_predicates"])
    diag_body = rules_res.get("diagnostics", rules_res)
    assert "coverage" in diag_body
    assert "dark_entities" in diag_body

    await mcp_harness.call(
        "remember",
        {
            "text": "SessionApi caches lookups in Memcached.",
            "triples": [
                {
                    "subject": "SessionApi",
                    "predicate": "uses_cache",
                    "object": "Memcached",
                }
            ],
        },
    )
    await mcp_harness.call(
        "remember",
        {
            "text": "SessionApi replaced Memcached with Valkey.",
            "triples": [
                {
                    "subject": "SessionApi",
                    "predicate": "uses_cache",
                    "object": "Valkey",
                }
            ],
        },
    )

    summary_delta = await mcp_harness.call(
        "changes_since",
        {"since": w0, "summary_only": True},
    )
    counts = summary_delta.get("counts") or summary_delta["changes"]["counts"]
    assert counts["new_relations"] >= 2
    assert counts["superseded_relations"] >= 1
    assert counts["new_documents"] >= 2

    detail_delta = await mcp_harness.call(
        "changes_since",
        {"since": w0, "summary_only": False, "limit": 50},
    )
    detail_body = detail_delta.get("changes") or detail_delta
    assert len(detail_body["new_relations"]) >= 2
    assert len(detail_body["superseded_relations"]) >= 1


async def test_realm_embedding_model_and_dimension_mismatch_raise_schema_error(
    mcp_harness_factory: Callable[..., Any],
) -> None:
    primary = mcp_harness_factory(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )
    shared_realm = primary.config.realm
    await primary.manager.get_rag()

    same_dim_wrong_model = mcp_harness_factory(
        realm=shared_realm,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    with pytest.raises(SchemaError, match="bge-small|all-MiniLM"):
        await same_dim_wrong_model.manager.get_rag()

    wrong_dim = mcp_harness_factory(
        realm=shared_realm,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=768,
    )
    with pytest.raises(SchemaError):
        await wrong_dim.manager.get_rag()
