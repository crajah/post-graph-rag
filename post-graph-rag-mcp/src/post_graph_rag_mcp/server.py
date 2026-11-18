"""FastMCP server exposing bi-temporal memory tools, resources, and prompts."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from post_graph_rag import (
    DocumentMetadata,
    Entity,
    ExtractionResult,
    GraphRAG,
    QueryParam,
    Triple,
)
from post_graph_rag.models import document_key
from pydantic import BaseModel, Field

from post_graph_rag_mcp import __version__
from post_graph_rag_mcp.config import MCPServerConfig
from post_graph_rag_mcp.engine_manager import EngineManager, extract_fallback_keywords
from post_graph_rag_mcp.ingest import OrderBy, run_ingest_documents

RecallMode = Literal["mix", "local", "global", "hybrid", "naive"]

SERVER_INSTRUCTIONS = (
    "Bi-temporal project knowledge graph. "
    "(1) Call `recall` (supplying `entity_keywords`) before proposing architectural, "
    "dependency, or schema changes. "
    "(2) Call `remember` with structured `triples` (setting `functional=True` on "
    "single-valued choices like `uses_auth`, `uses_database`, `deployed_on`) whenever "
    "a project convention, migration, or preference is stated or reversed. "
    "(3) Check `active_relations_on_subjects` in `remember`'s result; if an older "
    "synonymous edge stayed active, pass its ID in `supersedes_edge_ids` on a follow-up `remember` call."
)


class TripleInput(BaseModel):
    """Structured factual assertion supplied by the host agent."""

    subject: str = Field(
        ...,
        description="Canonical subject entity name (e.g. 'AuthService', 'Calder Industries').",
    )
    predicate: str = Field(
        ...,
        description="Active snake_case relation predicate (e.g. 'uses_auth', 'deployed_on', 'has_cfo').",
    )
    object: str = Field(
        ...,
        description="Canonical object entity name or scalar configuration value (e.g. 'BetterAuth', '8080').",
    )
    description: str | None = Field(
        default=None,
        description="Contextual detail or rationale supporting this relation.",
    )
    negated: bool = Field(
        default=False,
        description="Set True when the statement asserts that this relation does NOT hold.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Assertion confidence in [0.0, 1.0].",
    )
    valid_from: str | None = Field(
        default=None,
        description="Optional ISO-8601 date or year when world validity started (e.g. '2025-01-01').",
    )
    valid_to: str | None = Field(
        default=None,
        description="Optional ISO-8601 date or year when world validity ended.",
    )
    functional: bool = Field(
        default=False,
        description=(
            "Set True when this predicate is single-valued per subject so active prior "
            "edges with different objects are deterministically superseded."
        ),
    )


class EntityInput(BaseModel):
    """Canonical entity definition supplied by the host agent."""

    name: str = Field(..., description="Full canonical entity name.")
    type: str = Field(
        default="Concept",
        description="Entity category (e.g. 'Technology', 'Service', 'Person', 'Organization').",
    )
    description: str = Field(
        default="",
        description="Brief summary of the entity's role.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate names, abbreviations, or surface forms for this entity.",
    )


def _normalize_iso_instant(value: str | None) -> str | None:
    """Normalize ISO-8601 UTC instants (including trailing 'Z') to '+00:00' lexical form."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return cleaned
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _format_mcp_relation(rel: dict[str, Any]) -> dict[str, Any]:
    edge_id = (
        str(rel["edge_id"])
        if rel.get("edge_id") is not None
        else (str(rel["id"]) if rel.get("id") is not None else None)
    )
    subj = rel.get("subject") if rel.get("subject") is not None else rel.get("src_id")
    pred = rel.get("predicate") if rel.get("predicate") is not None else rel.get("relation_type")
    obj = rel.get("object") if rel.get("object") is not None else rel.get("tgt_id")
    superseded_by = (
        str(rel["superseded_by"]) if rel.get("superseded_by") is not None else None
    )
    return {
        "id": edge_id,
        "edge_id": edge_id,
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "src_id": subj,
        "relation_type": pred,
        "tgt_id": obj,
        "description": rel.get("description", ""),
        "negated": bool(rel.get("negated", False)),
        "confidence": float(rel.get("confidence", 1.0)),
        "weight": rel.get("weight", 1),
        "valid_from": rel.get("valid_from"),
        "valid_to": rel.get("valid_to"),
        "t_created": rel.get("t_created"),
        "t_expired": rel.get("t_expired"),
        "superseded_by": superseded_by,
        "asserted_at": rel.get("asserted_at"),
        "hops": rel.get("hops", 1),
    }


def _format_mcp_entity(ent: dict[str, Any]) -> dict[str, Any]:
    name = ent.get("name") or ent.get("entity_name")
    etype = ent.get("type") or ent.get("entity_type")
    return {
        "name": name,
        "type": etype,
        "entity_name": name,
        "entity_type": etype,
        "description": ent.get("description", ""),
    }


def _format_mcp_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = dict(chunk.get("metadata") or {})
    text = chunk.get("text") if chunk.get("text") is not None else chunk.get("content", "")
    return {
        "chunk_id": str(chunk["chunk_id"]) if chunk.get("chunk_id") is not None else None,
        "text": text,
        "content": text,
        "source": meta.get("source"),
        "document": meta.get("document"),
        "metadata": meta,
    }


async def _serialize_delta_edge(
    rag: GraphRAG,
    edge: Any,
    vertex_name_cache: dict[str, str],
) -> dict[str, Any]:
    if isinstance(edge, dict):
        return edge

    from_id = str(getattr(edge, "from_id", ""))
    to_id = str(getattr(edge, "to_id", ""))
    if from_id not in vertex_name_cache or to_id not in vertex_name_cache:
        src_v, tgt_v = await rag.store.get_relation_endpoints(edge)
        if src_v is not None:
            vertex_name_cache[from_id] = str((src_v.payload or {}).get("name") or from_id)
        if tgt_v is not None:
            vertex_name_cache[to_id] = str((tgt_v.payload or {}).get("name") or to_id)

    payload = dict(getattr(edge, "payload", None) or {})
    pred = getattr(edge, "relation_type", None)
    eid = str(getattr(edge, "id", ""))
    return {
        "id": eid,
        "edge_id": eid,
        "subject": vertex_name_cache.get(from_id, from_id),
        "predicate": pred,
        "object": vertex_name_cache.get(to_id, to_id),
        "from_id": from_id,
        "to_id": to_id,
        "description": payload.get("description", ""),
        "negated": bool(payload.get("negated", False)),
        "valid_from": payload.get("valid_from"),
        "valid_to": payload.get("valid_to"),
        "t_created": payload.get("t_created"),
        "t_expired": payload.get("t_expired"),
        "superseded_by": (
            str(payload["superseded_by"]) if payload.get("superseded_by") is not None else None
        ),
    }


def _serialize_delta_vertex(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        payload = item.get("payload") or {}
        return {
            "id": str(item.get("id", "")),
            "name": payload.get("name") or item.get("name"),
            "type": payload.get("type") or item.get("type"),
            "description": payload.get("description", ""),
            "dormant_since": payload.get("dormant_since"),
            "revived_at": payload.get("revived_at"),
            "created_at": item.get("created_at"),
        }
    payload = dict(getattr(item, "payload", None) or {})
    created_at = getattr(item, "created_at", None)
    return {
        "id": str(getattr(item, "id", "")),
        "name": payload.get("name"),
        "type": payload.get("type"),
        "description": payload.get("description", ""),
        "dormant_since": payload.get("dormant_since"),
        "revived_at": payload.get("revived_at"),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }


def _resolve_keywords(
    *,
    autonomous_llm_enabled: bool,
    question: str,
    entity_keywords: list[str] | None,
    theme_keywords: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    ll_kw = [k.strip() for k in (entity_keywords or []) if k and k.strip()]
    hl_kw = [k.strip() for k in (theme_keywords or []) if k and k.strip()]
    if not autonomous_llm_enabled and not ll_kw and not hl_kw:
        return extract_fallback_keywords(question)
    return (ll_kw, hl_kw)


def create_mcp_server(
    mcp_cfg: MCPServerConfig | None = None,
    manager: EngineManager | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> FastMCP:
    """Construct the ``post-graph-rag`` FastMCP server with 7 tools, 3 resources, and 3 prompts."""
    owns_manager = manager is None
    cfg = mcp_cfg or (manager.cfg if manager is not None else MCPServerConfig.from_env())
    mgr = manager if manager is not None else EngineManager(cfg)

    @asynccontextmanager
    async def _server_lifespan(_app: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_manager:
                await mgr.close()

    mcp = FastMCP(
        "post-graph-rag",
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        lifespan=_server_lifespan,
    )
    mcp._engine_manager = mgr  # type: ignore[attr-defined]
    mcp._mcp_config = cfg  # type: ignore[attr-defined]

    @mcp.tool()
    async def remember(
        text: str,
        triples: list[TripleInput] | None = None,
        entities: list[EntityInput] | None = None,
        source: str = "mcp:session",
        document: str | None = None,
        category: str | None = None,
        space: str | None = None,
        supersedes_edge_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store an auditable statement, architectural decision, or preference with supersession.

        Always supply structured ``triples`` when recording concrete facts so extraction runs
        in zero sidecar LLM calls. Set ``functional=True`` on single-valued predicates (e.g.
        ``uses_auth``, ``uses_database``, ``deployed_on``) so older objects are automatically
        retired. Inspect ``active_relations_on_subjects`` in the returned payload and pass
        ``supersedes_edge_ids`` on a follow-up call if an older synonymous edge stayed active.
        """
        if not text or not text.strip():
            raise ValueError("Parameter 'text' must be a non-empty string.")

        target_space = space or cfg.default_space
        doc_title = (
            document.strip()
            if (document and document.strip())
            else f"mem-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"
        )
        doc_key_val = document_key(source, doc_title)

        parsed_triples = [
            t if isinstance(t, TripleInput) else TripleInput.model_validate(t)
            for t in (triples or [])
        ]
        parsed_entities = [
            e if isinstance(e, EntityInput) else EntityInput.model_validate(e)
            for e in (entities or [])
        ]

        rag = await mgr.get_rag()

        dynamic_functional = [
            t.predicate for t in parsed_triples if t.functional and t.predicate.strip()
        ]
        if dynamic_functional:
            mgr.extend_rules(add_functional=dynamic_functional)

        ext_res: ExtractionResult | None
        if triples is not None or entities is not None:
            ext_res = ExtractionResult(
                entities=[
                    Entity(
                        name=e.name,
                        type=e.type,
                        description=e.description,
                        aliases=list(e.aliases),
                    )
                    for e in parsed_entities
                ],
                triples=[
                    Triple(
                        subject=t.subject,
                        predicate=t.predicate,
                        object=t.object,
                        description=t.description,
                        negated=t.negated,
                        confidence=t.confidence,
                        valid_from=t.valid_from,
                        valid_to=t.valid_to,
                    )
                    for t in parsed_triples
                ],
            )
        elif cfg.autonomous_llm_enabled:
            ext_res = None
        else:
            ext_res = ExtractionResult(entities=[], triples=[])

        meta = DocumentMetadata(
            source=source,
            document=doc_title,
            category=category,
            space=target_space,
        )
        idx_res = await rag.index_document(
            text,
            metadata=meta,
            space=target_space,
            extraction=ext_res,
        )

        added_edge_ids = [str(eid) for eid in (idx_res.get("added_edge_ids") or [])]
        superseded_details = list(idx_res.get("superseded_details") or [])
        covered_superseded = {
            str(d.get("id")) for d in superseded_details if isinstance(d, dict) and d.get("id")
        }
        for raw_sid in idx_res.get("superseded_ids") or []:
            sid_str = str(raw_sid)
            if sid_str not in covered_superseded:
                superseded_details.append({"id": sid_str})
                covered_superseded.add(sid_str)

        manually_retracted_ids: list[str] = []
        if supersedes_edge_ids:
            added_set = set(added_edge_ids)
            to_retract = [
                str(eid).strip()
                for eid in supersedes_edge_ids
                if str(eid).strip() and str(eid).strip() not in added_set
            ]
            if to_retract:
                closed_by_id = added_edge_ids[0] if added_edge_ids else "manual_retract"
                retracted = await rag.retract_relations(
                    to_retract, closed_by_id, space=target_space
                )
                manually_retracted_ids = [str(r) for r in retracted]

        subject_names: list[str] = []
        seen_subjects: set[str] = set()
        for t in parsed_triples:
            s_clean = t.subject.strip()
            if s_clean and s_clean.lower() not in seen_subjects:
                seen_subjects.add(s_clean.lower())
                subject_names.append(s_clean)
        if not subject_names:
            for ename in idx_res.get("entities") or []:
                e_clean = str(ename).strip()
                if e_clean and e_clean.lower() not in seen_subjects:
                    seen_subjects.add(e_clean.lower())
                    subject_names.append(e_clean)

        active_on_subjects = (
            await rag.active_relations_for_entities(
                subject_names,
                space=target_space,
                exclude_edge_ids=set(added_edge_ids),
                limit_per_entity=5,
            )
            if subject_names
            else {}
        )

        recorded_watermark = await rag.watermark(space=target_space)
        entities_written = max(
            int(idx_res.get("entities_extracted", 0)),
            int(idx_res.get("mentions_added", 0)),
        )

        return {
            "document_id": str(idx_res["document_id"]),
            "doc_key": doc_key_val,
            "recorded_at_watermark": recorded_watermark,
            "entities_written": entities_written,
            "relations_written": int(idx_res.get("relations_added", 0)),
            "added_edge_ids": added_edge_ids,
            "superseded_relations": superseded_details,
            "manually_retracted_ids": manually_retracted_ids,
            "active_relations_on_subjects": active_on_subjects,
        }

    @mcp.tool()
    async def recall(
        question: str,
        entity_keywords: list[str] | None = None,
        theme_keywords: list[str] | None = None,
        subqueries: list[str] | None = None,
        mode: RecallMode = "mix",
        top_k: int = 8,
        max_hops: int | None = None,
        space: str | None = None,
        synthesize: bool = False,
    ) -> dict[str, Any]:
        """Retrieve current truth from the knowledge graph (superseded relations excluded).

        Fuses multi-hop graph traversal, relation HNSW vector similarity, and PostgreSQL
        full-text search via Reciprocal Rank Fusion (RRF). Pass concrete ``entity_keywords``
        (service, component, or person names) to sharpen entity seeding without extra LLM calls.
        """
        if synthesize and not cfg.autonomous_llm_enabled:
            raise ValueError(
                "synthesize=True requires OPENAI_API_KEY (autonomous LLM mode). "
                "Leave synthesize=False (default) to return structured graph relations "
                "and passages directly to the host agent."
            )

        rag = await mgr.get_rag()
        target_space = space or cfg.default_space
        ll_kw, hl_kw = _resolve_keywords(
            autonomous_llm_enabled=cfg.autonomous_llm_enabled,
            question=question,
            entity_keywords=entity_keywords,
            theme_keywords=theme_keywords,
        )

        param = QueryParam(
            mode=mode,
            top_k=max(1, top_k),
            max_hops=max_hops,
            subqueries=subqueries,
            ll_keywords=ll_kw,
            hl_keywords=hl_kw,
            space=target_space,
            include_superseded=False,
        )

        data_res = await rag.query_data(question, param=param)
        data = data_res["data"]
        wm = await rag.watermark(space=target_space)

        result: dict[str, Any] = {
            "watermark": wm,
            "relationships": [_format_mcp_relation(r) for r in data.get("relationships", [])],
            "entities": [_format_mcp_entity(e) for e in data.get("entities", [])],
            "chunks": [_format_mcp_chunk(c) for c in data.get("chunks", [])],
            "communities": data.get("communities", []),
            "metadata": data_res.get("metadata", {}),
        }
        if synthesize:
            synth_res = await rag.query(question, param=param)
            if isinstance(synth_res, dict):
                result["answer"] = synth_res.get("answer")
        return result

    @mcp.tool()
    async def recall_as_of(
        question: str,
        as_of: str | None = None,
        as_believed_at: str | None = None,
        entity_keywords: list[str] | None = None,
        theme_keywords: list[str] | None = None,
        top_k: int = 8,
        space: str | None = None,
    ) -> dict[str, Any]:
        """Bi-temporal time-travel & audit query over valid time and/or belief time.

        Supply ``as_of`` (e.g. ``'2024-06-01'``) to query facts valid in the world on that
        date, and/or ``as_believed_at`` (an ISO-8601 ``recorded_at_watermark``) to audit
        what this memory store believed at that transaction instant prior to later restatements.
        """
        norm_as_of = as_of.strip() if (as_of and as_of.strip()) else None
        norm_believed = _normalize_iso_instant(as_believed_at)
        if norm_as_of is None and norm_believed is None:
            raise ValueError(
                "recall_as_of requires at least one temporal filter: pass 'as_of' "
                "(world validity date) and/or 'as_believed_at' (ISO-8601 belief watermark)."
            )

        rag = await mgr.get_rag()
        target_space = space or cfg.default_space
        ll_kw, hl_kw = _resolve_keywords(
            autonomous_llm_enabled=cfg.autonomous_llm_enabled,
            question=question,
            entity_keywords=entity_keywords,
            theme_keywords=theme_keywords,
        )

        param = QueryParam(
            mode="mix",
            top_k=max(1, top_k),
            as_of=norm_as_of,
            as_believed_at=norm_believed,
            ll_keywords=ll_kw,
            hl_keywords=hl_kw,
            space=target_space,
        )

        data_res = await rag.query_data(question, param=param)
        data = data_res["data"]
        wm = await rag.watermark(space=target_space)

        return {
            "watermark": wm,
            "temporal_axes_applied": {
                "as_of": norm_as_of,
                "as_believed_at": as_believed_at.strip() if as_believed_at else None,
            },
            "relationships": [_format_mcp_relation(r) for r in data.get("relationships", [])],
            "entities": [_format_mcp_entity(e) for e in data.get("entities", [])],
            "chunks": [_format_mcp_chunk(c) for c in data.get("chunks", [])],
            "communities": data.get("communities", []),
            "metadata": data_res.get("metadata", {}),
        }

    @mcp.tool()
    async def trace_history(
        question: str,
        entity_keywords: list[str] | None = None,
        top_k: int = 12,
        space: str | None = None,
    ) -> dict[str, Any]:
        """Trace full chronological evolution and supersession lineage (outbound and inbound).

        Returns both active ``current_relations`` and retired ``superseded_relations`` sorted
        chronologically by ``(valid_from, t_created)``, walking edges in both directions so
        object-side questions (e.g. 'Where was Clerk used?') surface retired dependencies.
        """
        rag = await mgr.get_rag()
        target_space = space or cfg.default_space
        ll_kw, hl_kw = _resolve_keywords(
            autonomous_llm_enabled=cfg.autonomous_llm_enabled,
            question=question,
            entity_keywords=entity_keywords,
            theme_keywords=[question.strip()] if (not cfg.autonomous_llm_enabled and question.strip()) else None,
        )

        qp_kwargs: dict[str, Any] = {
            "mode": "mix",
            "include_superseded": True,
            "top_k": max(1, top_k),
            "ll_keywords": ll_kw,
            "hl_keywords": hl_kw,
            "space": target_space,
        }
        if "direction" in inspect.signature(QueryParam.__init__).parameters:
            qp_kwargs["direction"] = "both"

        data_res = await rag.query_data(question, param=QueryParam(**qp_kwargs))
        data = data_res["data"]

        by_edge_key: dict[str, dict[str, Any]] = {}
        fallback_seq = 0
        for raw_rel in data.get("relationships", []):
            formatted = _format_mcp_relation(raw_rel)
            key = formatted.get("id") or f"_seq_{fallback_seq}"
            fallback_seq += 1
            by_edge_key[key] = formatted

        # Ensure multi-epoch reverts (A -> B -> A) and inbound object-side edges on
        # matched entities are never collapsed by (src, pred, tgt) deduplication.
        candidate_names: list[str] = []
        seen_names: set[str] = set()
        for kw in entity_keywords or []:
            if kw and kw.strip() and kw.strip().lower() not in seen_names:
                seen_names.add(kw.strip().lower())
                candidate_names.append(kw.strip())
        for ent in data.get("entities", []):
            ename = str(ent.get("entity_name") or ent.get("name") or "").strip()
            if ename and ename.lower() not in seen_names:
                seen_names.add(ename.lower())
                candidate_names.append(ename)

        nb_params = inspect.signature(rag.store.get_neighborhood).parameters
        nb_extra: dict[str, Any] = {"direction": "both"} if "direction" in nb_params else {}
        for cname in candidate_names[: max(4, top_k)]:
            vertex = await rag.store.find_entity(cname, space=target_space)
            if vertex is None:
                continue
            for edge, src_v, tgt_v, hops in await rag.store.get_neighborhood(
                vertex.id,
                max_hops=1,
                space=target_space,
                include_superseded=True,
                max_edges=max(50, top_k * 4),
                **nb_extra,
            ):
                raw_triple = rag._format_triple(edge, src_v, tgt_v, hops=hops)
                formatted = _format_mcp_relation(raw_triple)
                key = formatted.get("id") or f"_nb_{fallback_seq}"
                fallback_seq += 1
                by_edge_key.setdefault(key, formatted)

        def _chrono_key(rel: dict[str, Any]) -> tuple[str, str, str]:
            return (
                str(rel.get("valid_from") or ""),
                str(rel.get("t_created") or ""),
                str(rel.get("id") or ""),
            )

        all_relations = sorted(by_edge_key.values(), key=_chrono_key)
        current_relations = [r for r in all_relations if r.get("superseded_by") is None]
        superseded_relations = [r for r in all_relations if r.get("superseded_by") is not None]
        wm = await rag.watermark(space=target_space)

        return {
            "watermark": wm,
            "current_relations": current_relations,
            "superseded_relations": superseded_relations,
            "timeline": all_relations,
            "total_epochs": len(all_relations),
            "entities": [_format_mcp_entity(e) for e in data.get("entities", [])],
            "chunks": [_format_mcp_chunk(c) for c in data.get("chunks", [])],
        }

    @mcp.tool()
    async def ingest_documents(
        paths: list[str],
        glob: str = "**/*.md",
        order_by: OrderBy = "filename",
        collection: str | None = None,
        space: str | None = None,
        build_communities: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Idempotently sync Markdown / ADR files from allowed workspace paths into memory.

        Deterministically orders files by ``filename`` (default, e.g. ``0001-clerk.md`` before
        ``0002-betterauth.md``), ``mtime``, or ``git_first_commit`` so newer decisions supersede
        older ones, and skips unchanged files via SHA-256 content hashes.
        """
        return await run_ingest_documents(
            mcp_cfg=cfg,
            manager=mgr,
            paths=paths,
            glob=glob,
            order_by=order_by,
            collection=collection,
            space=space,
            build_communities=build_communities,
            ctx=ctx,
        )

    @mcp.tool()
    async def changes_since(
        since: str | None = None,
        space: str | None = None,
        summary_only: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Poll database-clock belief deltas since an ISO-8601 watermark.

        Call with ``since=None`` to obtain the initial database ``watermark``, then pass that
        watermark as ``since`` on subsequent calls to inspect newly added or superseded facts.
        """
        rag = await mgr.get_rag()
        target_space = space or cfg.default_space
        if since is None or not since.strip():
            return {
                "watermark": await rag.watermark(space=target_space),
                "changes": None,
            }

        norm_since = _normalize_iso_instant(since) or since.strip()
        delta = await rag.changes_since(
            norm_since,
            space=target_space,
            summary=summary_only,
            limit=max(1, limit),
        )

        vertex_cache: dict[str, str] = {}
        new_rels = [
            await _serialize_delta_edge(rag, edge, vertex_cache)
            for edge in delta.new_relations
        ]
        sup_rels = [
            await _serialize_delta_edge(rag, edge, vertex_cache)
            for edge in delta.superseded_relations
        ]
        new_ents = [_serialize_delta_vertex(v) for v in delta.new_entities]
        dor_ents = [_serialize_delta_vertex(v) for v in delta.dormant_entities]
        rev_ents = [_serialize_delta_vertex(v) for v in delta.revived_entities]
        new_docs = list(delta.new_documents)

        changes_payload: dict[str, Any] = {
            "since": delta.since,
            "as_of": delta.as_of,
            "watermark": delta.as_of,
            "empty": delta.empty,
            "counts": dict(delta.counts),
            "communities_stale": delta.communities_stale,
            "new_relations": new_rels,
            "superseded_relations": sup_rels,
            "new_entities": new_ents,
            "dormant_entities": dor_ents,
            "revived_entities": rev_ents,
            "new_documents": new_docs,
        }

        result: dict[str, Any] = {
            "watermark": delta.as_of,
            "since": delta.since,
            "empty": delta.empty,
            "counts": dict(delta.counts),
            "communities_stale": delta.communities_stale,
            "changes": changes_payload,
        }
        if not summary_only:
            result.update(
                {
                    "new_relations": new_rels,
                    "superseded_relations": sup_rels,
                    "new_entities": new_ents,
                    "dormant_entities": dor_ents,
                    "revived_entities": rev_ents,
                    "new_documents": new_docs,
                }
            )
        return result

    @mcp.tool()
    async def configure_memory_rules(
        preset: str | None = None,
        add_functional_predicates: list[str] | None = None,
        add_exclusive_groups: list[list[str]] | None = None,
        include_diagnostics: bool = True,
        space: str | None = None,
    ) -> dict[str, Any]:
        """Inspect or extend active supersession presets, functional predicates, and diagnostics."""
        rules = mgr.extend_rules(
            preset=preset,
            add_functional=add_functional_predicates,
            add_exclusive_groups=add_exclusive_groups,
        )
        if not include_diagnostics:
            return {**rules, "diagnostics": None}

        rag = await mgr.get_rag()
        target_space = space or cfg.default_space

        coverage_rows = [dataclasses.asdict(c) for c in await rag.coverage(space=target_space)]
        dark_rows = await rag.dark_entities(space=target_space)
        doc_delta = await rag.changes_since(
            "1970-01-01T00:00:00+00:00",
            space=target_space,
            include=("documents",),
            summary=False,
            limit=200,
        )
        doc_keys = [
            str(d["doc_key"]) for d in doc_delta.new_documents if d.get("doc_key")
        ]
        doc_stats_map = await rag.documents_stats(doc_keys, space=target_space)
        serialized_doc_stats = {k: v.to_dict() for k, v in doc_stats_map.items()}

        diagnostics = {
            "space": target_space,
            "watermark": doc_delta.as_of,
            "coverage": coverage_rows,
            "dark_entities": dark_rows,
            "documents_stats": serialized_doc_stats,
        }
        return {
            **rules,
            "coverage": coverage_rows,
            "dark_entities": dark_rows,
            "documents_stats": serialized_doc_stats,
            "diagnostics": diagnostics,
        }

    # --------------------------------------------------------------- resources

    @mcp.resource(
        "memory://status",
        name="memory_status",
        description="Active realm, embedding fingerprint, corpus counts, and database watermark.",
        mime_type="application/json",
    )
    async def resource_memory_status() -> str:
        rag = await mgr.get_rag()
        target_space = cfg.default_space
        delta = await rag.changes_since(
            "1970-01-01T00:00:00+00:00",
            space=target_space,
            summary=True,
        )
        counts = dict(delta.counts)
        payload = {
            "server_version": __version__,
            "realm": cfg.realm,
            "space": target_space,
            "preset": mgr.preset_name,
            "storage_mode": "external_postgres" if cfg.postgres_uri else "embedded_pgserver",
            "embedding_backend": cfg.embedding_backend,
            "embedding_model": cfg.embedding_model,
            "embedding_dim": cfg.embedding_dim,
            "autonomous_llm_enabled": cfg.autonomous_llm_enabled,
            "contradiction_detection": cfg.contradiction_detection,
            "watermark": delta.as_of,
            "counts": counts,
            "active_entities": max(
                0, counts.get("new_entities", 0) - counts.get("dormant_entities", 0)
            ),
            "active_relations": max(
                0, counts.get("new_relations", 0) - counts.get("superseded_relations", 0)
            ),
            "superseded_relations": counts.get("superseded_relations", 0),
            "documents": counts.get("new_documents", 0),
        }
        return json.dumps(payload, indent=2)

    @mcp.resource(
        "memory://rules",
        name="memory_rules",
        description="Active functional predicates, exclusive predicate groups, and aliases.",
        mime_type="application/json",
    )
    async def resource_memory_rules() -> str:
        return json.dumps(mgr.get_active_rules(), indent=2)

    @mcp.resource(
        "memory://communities",
        name="memory_communities",
        description="Hierarchical community report tree for the active realm and space.",
        mime_type="application/json",
    )
    async def resource_memory_communities() -> str:
        rag = await mgr.get_rag()
        tree = await rag.get_community_tree(space=cfg.default_space)
        return json.dumps(
            {
                "realm": cfg.realm,
                "space": cfg.default_space,
                **tree,
            },
            indent=2,
        )

    # ----------------------------------------------------------------- prompts

    @mcp.prompt(
        name="remember-architecture-decision",
        description="Extract structured entities and triples from an architectural decision and persist via remember.",
    )
    async def prompt_remember_architecture_decision(
        decision_summary: str = "",
        superseded_choice: str = "",
    ) -> str:
        context_line = (
            f"Decision context to record:\n{decision_summary}\n\n"
            if decision_summary.strip()
            else ""
        )
        superseded_line = (
            f"Previously active choice being replaced: {superseded_choice}\n\n"
            if superseded_choice.strip()
            else ""
        )
        return (
            f"{context_line}{superseded_line}"
            "Persist this architectural decision into project memory using the `remember` tool:\n"
            "1. Extract canonical `entities` (`name`, `type` such as 'Service' or 'Technology', "
            "`description`, and `aliases`).\n"
            "2. Extract concrete `triples` (`subject`, `predicate`, `object`, `description`, "
            "optional `valid_from`). Prefer standard functional predicates (`uses_auth`, "
            "`uses_database`, `uses_framework`, `uses_orm`, `uses_queue`, `deployed_on`, "
            "`port_is`, `api_version`) and set `functional=True` for single-valued slots.\n"
            "3. Call `remember` with `category='architecture'`.\n"
            "4. Inspect `active_relations_on_subjects` in the response. If an older synonymous "
            "predicate for the replaced technology remained active, call `remember` with "
            "`supersedes_edge_ids` to close it cleanly."
        )

    @mcp.prompt(
        name="audit-belief-state",
        description="Compare current project truth against historical valid-time and belief-time checkpoints.",
    )
    async def prompt_audit_belief_state(
        topic: str = "",
        checkpoint_timestamp: str = "",
        valid_date: str = "",
    ) -> str:
        subject_focus = topic.strip() or "the target system or entity"
        steps = [
            f"Run a bi-temporal belief and validity audit for {subject_focus}:",
            "1. Call `recall` with specific `entity_keywords` to retrieve the currently active truth.",
            "2. Call `trace_history` with the same `entity_keywords` to inspect all superseded epochs in chronological order.",
        ]
        if checkpoint_timestamp.strip():
            steps.append(
                f"3. Call `recall_as_of` with `as_believed_at='{checkpoint_timestamp.strip()}'` "
                "to reconstruct exactly what the knowledge graph believed at that transaction watermark."
            )
        if valid_date.strip():
            steps.append(
                f"4. Call `recall_as_of` with `as_of='{valid_date.strip()}'` "
                "to inspect which facts held in the world on that date."
            )
        steps.append(
            "Summarize what changed, when each belief epoch was recorded or superseded, and cite supporting documents."
        )
        return "\n".join(steps)

    @mcp.prompt(
        name="sync-adr-folder",
        description="Sync an ADR or Markdown architecture folder in chronological order with host-extracted triples.",
    )
    async def prompt_sync_adr_folder(folder_path: str = "docs/adr") -> str:
        target = folder_path.strip() or "docs/adr"
        return (
            f"Synchronize all Architecture Decision Records in `{target}` into bi-temporal memory "
            "(Zero-Key Host-Agent workflow):\n"
            f"1. List all Markdown files under `{target}` sorted in ascending lexicographic filename order "
            "(e.g. `0001-...md` before `0002-...md`) so later decisions supersede earlier ones.\n"
            "2. Read each ADR in order. Extract canonical `entities` and factual `triples` "
            "(setting `functional=True` on single-valued choices such as `uses_auth`, `uses_database`, "
            "`deployed_on`, and setting `valid_from` only when the ADR explicitly states a date).\n"
            "3. Call `remember` once per ADR passing `source=<relative_adr_path>`, `document=<adr_title>`, "
            "`category='architecture'`, `entities`, and `triples`.\n"
            "4. Check `active_relations_on_subjects` after each call; if a later ADR used a different "
            "predicate to retire an earlier ADR's choice, pass the old edge ID in `supersedes_edge_ids`.\n"
            "5. Finish by calling `trace_history` on the core services to confirm superseded decisions "
            "are closed and current choices are active."
        )

    return mcp


__all__ = [
    "EntityInput",
    "SERVER_INSTRUCTIONS",
    "TripleInput",
    "create_mcp_server",
]
