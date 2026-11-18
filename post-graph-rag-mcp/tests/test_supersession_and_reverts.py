"""Live-PostgreSQL tests for deterministic functional supersession, reverts, polarity flips, and manual retraction."""
from __future__ import annotations

from typing import Any, Callable


async def test_engineering_preset_functional_supersession(
    mcp_harness: Any,
) -> None:
    first = await mcp_harness.call(
        "remember",
        {
            "text": "AuthService uses Clerk for user authentication.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "Clerk",
                }
            ],
        },
    )
    assert first["relations_written"] == 1
    assert first["superseded_relations"] == []

    second = await mcp_harness.call(
        "remember",
        {
            "text": "AuthService migrated off Clerk and now uses BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "BetterAuth",
                }
            ],
        },
    )
    assert second["relations_written"] == 1
    assert any(mcp_harness.old_object(s) == "Clerk" for s in second["superseded_relations"])

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "Which authentication provider does AuthService use?",
            "entity_keywords": ["AuthService"],
        },
    )
    auth_objects = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    assert auth_objects == ["BetterAuth"]


async def test_predicate_alias_snapping_closes_prior_synonym(
    mcp_harness: Any,
) -> None:
    await mcp_harness.call(
        "remember",
        {
            "text": "AuthService relies on Clerk as its auth provider.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "auth_provider",
                    "object": "Clerk",
                }
            ],
        },
    )
    second = await mcp_harness.call(
        "remember",
        {
            "text": "AuthService switched its auth provider to BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth_provider",
                    "object": "BetterAuth",
                }
            ],
        },
    )
    assert any(mcp_harness.old_object(s) == "Clerk" for s in second["superseded_relations"])

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "What authentication provider is active on AuthService?",
            "entity_keywords": ["AuthService"],
        },
    )
    active_objects = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    assert active_objects == ["BetterAuth"]


async def test_general_memory_tense_prefix_normalization_closes_prior_role(
    mcp_harness_factory: Callable[..., Any],
) -> None:
    harness = mcp_harness_factory(preset="general_memory")

    await harness.call(
        "remember",
        {
            "text": "Alice holds the role of Engineer.",
            "triples": [
                {
                    "subject": "Alice",
                    "predicate": "has_role",
                    "object": "Engineer",
                }
            ],
        },
    )
    second = await harness.call(
        "remember",
        {
            "text": "Alice was promoted to CTO.",
            "triples": [
                {
                    "subject": "Alice",
                    "predicate": "has_role",
                    "object": "CTO",
                }
            ],
        },
    )
    assert any(harness.old_object(s) == "Engineer" for s in second["superseded_relations"])

    recalled = await harness.call(
        "recall",
        {
            "question": "What is Alice's current role?",
            "entity_keywords": ["Alice"],
        },
    )
    roles = [
        harness.object(r)
        for r in recalled["relationships"]
        if harness.subject(r) == "Alice"
    ]
    assert roles == ["CTO"]


async def test_numeric_scalar_objects_survive_validation_and_supersede_on_functional_predicate(
    mcp_harness: Any,
) -> None:
    first = await mcp_harness.call(
        "remember",
        {
            "text": "ApiServer listens on port 8080.",
            "triples": [
                {
                    "subject": "ApiServer",
                    "predicate": "port_is",
                    "object": "8080",
                }
            ],
        },
    )
    assert first["relations_written"] == 1

    second = await mcp_harness.call(
        "remember",
        {
            "text": "ApiServer moved from port 8080 to port 9090.",
            "triples": [
                {
                    "subject": "ApiServer",
                    "predicate": "port_is",
                    "object": "9090",
                }
            ],
        },
    )
    assert second["relations_written"] == 1
    assert any(mcp_harness.old_object(s) == "8080" for s in second["superseded_relations"])

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "Which port does ApiServer listen on?",
            "entity_keywords": ["ApiServer"],
        },
    )
    ports = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "ApiServer"
    ]
    assert ports == ["9090"]


async def test_dynamic_functional_flag_on_custom_domain_predicate(
    mcp_harness: Any,
) -> None:
    await mcp_harness.call(
        "remember",
        {
            "text": "Checkout settles invoices in USD.",
            "triples": [
                {
                    "subject": "Checkout",
                    "predicate": "billing_currency",
                    "object": "USD",
                    "functional": True,
                }
            ],
        },
    )
    second = await mcp_harness.call(
        "remember",
        {
            "text": "Checkout changed settlement currency from USD to EUR.",
            "triples": [
                {
                    "subject": "Checkout",
                    "predicate": "billing_currency",
                    "object": "EUR",
                    "functional": True,
                }
            ],
        },
    )
    assert any(mcp_harness.old_object(s) == "USD" for s in second["superseded_relations"])

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "What billing currency does Checkout use?",
            "entity_keywords": ["Checkout"],
        },
    )
    currencies = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "Checkout"
    ]
    assert currencies == ["EUR"]


async def test_a_to_b_to_a_revert_creates_three_distinct_belief_epochs(
    mcp_harness: Any,
) -> None:
    e1 = await mcp_harness.call(
        "remember",
        {
            "text": "Epoch 1: AuthService uses Clerk.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "Clerk",
                    "valid_from": "2024-01-01",
                    "valid_to": "2024-06-30",
                }
            ],
        },
    )
    e2 = await mcp_harness.call(
        "remember",
        {
            "text": "Epoch 2: AuthService switched to BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "BetterAuth",
                    "valid_from": "2024-07-01",
                    "valid_to": "2024-12-31",
                }
            ],
        },
    )
    e3 = await mcp_harness.call(
        "remember",
        {
            "text": "Epoch 3: AuthService reverted from BetterAuth back to Clerk.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "Clerk",
                    "valid_from": "2025-01-01",
                }
            ],
        },
    )

    assert len({e1["added_edge_ids"][0], e2["added_edge_ids"][0], e3["added_edge_ids"][0]}) == 3
    assert any(mcp_harness.old_object(s) == "BetterAuth" for s in e3["superseded_relations"])

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "What authentication provider does AuthService currently use?",
            "entity_keywords": ["AuthService"],
        },
    )
    active_objs = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    assert active_objs == ["Clerk"]

    history = await mcp_harness.call(
        "trace_history",
        {
            "question": "Trace AuthService authentication migrations.",
            "entity_keywords": ["AuthService"],
            "top_k": 12,
        },
    )
    curr = [
        r
        for r in history["current_relations"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    sup = [
        r
        for r in history["superseded_relations"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    assert len(curr) == 1
    assert mcp_harness.object(curr[0]) == "Clerk"
    assert len(sup) == 2
    assert {mcp_harness.object(r) for r in sup} == {"Clerk", "BetterAuth"}


async def test_polarity_flip_supersedes_positive_edge_and_activates_negated_edge(
    mcp_harness: Any,
) -> None:
    pos = await mcp_harness.call(
        "remember",
        {
            "text": "App is deployed on Fly.io.",
            "triples": [
                {
                    "subject": "App",
                    "predicate": "deployed_on",
                    "object": "Fly.io",
                    "negated": False,
                }
            ],
        },
    )
    neg = await mcp_harness.call(
        "remember",
        {
            "text": "App is no longer deployed on Fly.io.",
            "triples": [
                {
                    "subject": "App",
                    "predicate": "deployed_on",
                    "object": "Fly.io",
                    "negated": True,
                }
            ],
        },
    )
    assert pos["added_edge_ids"][0] != neg["added_edge_ids"][0]
    assert len(neg["superseded_relations"]) == 1

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "Is App deployed on Fly.io?",
            "entity_keywords": ["App"],
        },
    )
    app_edges = [
        r
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "App" and mcp_harness.object(r) == "Fly.io"
    ]
    assert len(app_edges) == 1
    assert app_edges[0]["negated"] is True


async def test_zero_llm_host_contradiction_loop_via_active_relations_and_supersedes_edge_ids(
    mcp_harness: Any,
) -> None:
    first = await mcp_harness.call(
        "remember",
        {
            "text": "AuthService integrates identity via Clerk.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "custom_old_pred",
                    "object": "Clerk",
                    "functional": False,
                }
            ],
        },
    )
    clerk_edge_id = str(first["added_edge_ids"][0])

    second = await mcp_harness.call(
        "remember",
        {
            "text": "AuthService authenticates sessions via BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "custom_new_pred",
                    "object": "BetterAuth",
                    "functional": False,
                }
            ],
        },
    )
    active_map = second["active_relations_on_subjects"]
    subject_edges = active_map.get("AuthService", [])
    exposed_ids = {mcp_harness.edge_id(r) for r in subject_edges}
    assert clerk_edge_id in exposed_ids

    third = await mcp_harness.call(
        "remember",
        {
            "text": "Retire legacy Clerk custom_old_pred relation in favor of BetterAuth.",
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "custom_new_pred",
                    "object": "BetterAuth",
                    "functional": False,
                }
            ],
            "supersedes_edge_ids": [clerk_edge_id],
        },
    )
    assert clerk_edge_id in [str(x) for x in third["manually_retracted_ids"]]

    recalled = await mcp_harness.call(
        "recall",
        {
            "question": "Which authentication service is active for AuthService?",
            "entity_keywords": ["AuthService"],
        },
    )
    remaining_objs = [
        mcp_harness.object(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    ]
    assert "BetterAuth" in remaining_objs
    assert "Clerk" not in remaining_objs
    assert all(
        "custom_old_pred" not in mcp_harness.predicate(r)
        for r in recalled["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    )
