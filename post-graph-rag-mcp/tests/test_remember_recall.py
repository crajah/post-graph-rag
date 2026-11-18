"""Live-PostgreSQL tests for zero-key remember/recall, constant-source isolation, and FastEmbed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import pytest
from post_graph_rag import RAGConfig
from post_graph_rag.errors import LLMError

from post_graph_rag_mcp.config import MCPServerConfig, derive_workspace_realm
from post_graph_rag_mcp.embeddings import FastEmbedLLMService, build_llm_service
from post_graph_rag_mcp.engine_manager import extract_fallback_keywords
from post_graph_rag_mcp.presets import get_preset, list_presets


def test_workspace_realm_derivation_and_presets(tmp_path: Path) -> None:
    repo_dir = tmp_path / "Billing-Core.Service"
    nested = repo_dir / "services" / "invoicing"
    nested.mkdir(parents=True)
    (repo_dir / ".git").mkdir()

    realm = derive_workspace_realm(nested)
    assert realm.startswith("repo_")
    assert "billing_core_service" in realm
    assert len(realm) <= 40

    presets = list_presets()
    assert "engineering" in presets
    assert "general_memory" in presets

    eng = get_preset("engineering")
    assert len(eng.functional_predicates) >= 15
    assert len(eng.exclusive_predicate_groups) >= 4
    assert len(eng.predicate_aliases) >= 8

    with pytest.raises(ValueError):
        get_preset("nonexistent_preset_name")

    ll_kw, hl_kw = extract_fallback_keywords(
        "Which authentication provider does AuthService rely on?"
    )
    assert any("authservice" in k.lower() for k in ll_kw)
    assert len(hl_kw) >= 1


async def test_zero_key_remember_and_rrf_recall_with_and_without_keywords(
    mcp_harness: Any,
) -> None:
    write_res = await mcp_harness.call(
        "remember",
        {
            "text": (
                "AuthService uses BetterAuth for session tokens and PostgreSQL "
                "for transactional user storage."
            ),
            "entities": [
                {
                    "name": "AuthService",
                    "type": "Technology",
                    "description": "Primary authentication service",
                },
                {
                    "name": "BetterAuth",
                    "type": "Technology",
                    "description": "TypeScript authentication library",
                },
                {
                    "name": "PostgreSQL",
                    "type": "Technology",
                    "description": "Primary relational database engine",
                },
            ],
            "triples": [
                {
                    "subject": "AuthService",
                    "predicate": "uses_auth",
                    "object": "BetterAuth",
                    "description": "Handles user sessions and OAuth",
                },
                {
                    "subject": "AuthService",
                    "predicate": "uses_database",
                    "object": "PostgreSQL",
                    "description": "Persists account credentials and sessions",
                },
            ],
        },
    )

    assert write_res["document_id"] is not None
    assert write_res["recorded_at_watermark"]
    assert write_res["entities_written"] >= 3
    assert write_res["relations_written"] == 2
    assert len(write_res["added_edge_ids"]) == 2

    # 1. Explicit entity_keywords + theme_keywords RRF recall
    explicit_res = await mcp_harness.call(
        "recall",
        {
            "question": "How does AuthService manage sessions and storage?",
            "entity_keywords": ["AuthService"],
            "theme_keywords": ["authentication", "database"],
            "top_k": 8,
        },
    )
    assert explicit_res["watermark"]
    objects_found = {
        mcp_harness.object(r)
        for r in explicit_res["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    }
    assert {"BetterAuth", "PostgreSQL"} <= objects_found
    assert any("BetterAuth" in mcp_harness.chunk_text(c) for c in explicit_res["chunks"])
    assert mcp_harness.total_chat_calls == 0

    # 2. Empty keyword lists -> Mode-A deterministic extract_fallback_keywords (0 chat LLM calls)
    fallback_res = await mcp_harness.call(
        "recall",
        {
            "question": "Which authentication library does AuthService use?",
            "entity_keywords": [],
            "theme_keywords": [],
            "top_k": 8,
        },
    )
    fallback_objects = {
        mcp_harness.object(r)
        for r in fallback_res["relationships"]
        if mcp_harness.subject(r) == "AuthService"
    }
    assert "BetterAuth" in fallback_objects
    assert mcp_harness.total_chat_calls == 0

    # 3. Mode-A synthesize=True guard without an autonomous chat LLM key
    with pytest.raises(ValueError, match="OPENAI_API_KEY|synthesize"):
        await mcp_harness.call(
            "recall",
            {
                "question": "Summarize AuthService dependencies.",
                "entity_keywords": ["AuthService"],
                "synthesize": True,
            },
        )
    assert mcp_harness.total_chat_calls == 0


async def test_constant_mcp_session_source_keeps_five_passages_distinct(
    mcp_harness: Any,
) -> None:
    service_facts = [
        ("GatewayService", "deployed_on", "CloudflareWorkers", "Edge API routing"),
        ("BillingService", "uses_database", "PostgreSQL", "Ledger storage"),
        ("WorkerService", "uses_queue", "BullMQ", "Async job dispatch"),
        ("WebFrontend", "uses_framework", "NextJS", "SSR web application"),
        ("MonorepoBuild", "uses_package_manager", "pnpm", "Workspace package tool"),
    ]

    doc_keys: list[str] = []
    doc_ids: list[str] = []

    for subj, pred, obj, note in service_facts:
        res = await mcp_harness.call(
            "remember",
            {
                "text": f"{subj} standardized on {obj} ({note}).",
                "source": "mcp:session",
                "document": None,
                "triples": [
                    {
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "description": note,
                    }
                ],
            },
        )
        doc_keys.append(res["doc_key"])
        doc_ids.append(str(res["document_id"]))

    assert len(set(doc_keys)) == 5
    assert len(set(doc_ids)) == 5
    assert all("mem-" in dk for dk in doc_keys)

    rag = await mcp_harness.manager.get_rag()
    stats = await rag.documents_stats(doc_keys)
    assert len(stats) == 5
    for st in stats.values():
        assert st.found is True
        assert st.chunks == 1
        assert st.entities_dormant == 0

    for subj, _pred, obj, _note in service_facts:
        recalled = await mcp_harness.call(
            "recall",
            {
                "question": f"What technology is {subj} standardized on?",
                "entity_keywords": [subj],
                "top_k": 8,
            },
        )
        matched = [
            r
            for r in recalled["relationships"]
            if mcp_harness.subject(r) == subj and mcp_harness.object(r) == obj
        ]
        assert len(matched) == 1, f"Expected active relation ({subj} -> {obj})"
        assert any(subj in mcp_harness.chunk_text(c) for c in recalled["chunks"])


@pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is None,
    reason="fastembed optional extra is not installed",
)
async def test_live_fastembed_llm_service_end_to_end(
    mcp_harness_factory: Callable[..., Any],
    clean_mcp_env: Path,
) -> None:
    mcp_cfg = MCPServerConfig.from_env(cwd=clean_mcp_env)
    rag_cfg = RAGConfig(
        db_uri=mcp_cfg.postgres_uri or "postgresql://localhost:5432/postgres",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )
    svc = build_llm_service(mcp_cfg, rag_cfg)
    assert isinstance(svc, FastEmbedLLMService)
    assert getattr(svc, "_embedder", None) is None

    with pytest.raises(LLMError):
        await svc.chat_completion([{"role": "user", "content": "ping"}])

    harness = mcp_harness_factory(
        use_real_fastembed=True,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
    )
    await harness.call(
        "remember",
        {
            "text": "SearchService indexes product embeddings in pgvector HNSW tables.",
            "triples": [
                {
                    "subject": "SearchService",
                    "predicate": "uses_database",
                    "object": "pgvector",
                    "description": "Vector similarity indexing",
                }
            ],
        },
    )
    recalled = await harness.call(
        "recall",
        {
            "question": "Where does SearchService store vector embeddings?",
            "entity_keywords": ["SearchService"],
        },
    )
    assert any(
        harness.subject(r) == "SearchService"
        and harness.object(r) == "pgvector"
        and "database" in harness.predicate(r)
        for r in recalled["relationships"]
    )
