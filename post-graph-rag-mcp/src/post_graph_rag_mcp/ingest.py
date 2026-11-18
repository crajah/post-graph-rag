"""Sandboxed Markdown/ADR directory ingestion with frontmatter triples and deterministic ordering."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import Context
from mcp.types import ClientCapabilities, SamplingCapability, SamplingMessage, TextContent
from post_graph_rag import (
    DocumentContext,
    DocumentMetadata,
    Entity,
    ExtractionResult,
    GraphRAG,
    Triple,
)
from post_graph_rag.extractor import GraphExtractor
from post_graph_rag.models import content_hash, document_key

from post_graph_rag_mcp.config import MCPServerConfig
from post_graph_rag_mcp.engine_manager import EngineManager

logger = logging.getLogger(__name__)

OrderBy = Literal["filename", "mtime", "git_first_commit"]
ExtractionMode = Literal["llm_graph", "mcp_sampling", "chunks_and_frontmatter"]

_FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

_SAMPLING_SYSTEM_PROMPT = (
    "Extract factual entities and (subject, predicate, object) triples from the passage. "
    "Return ONLY valid JSON matching "
    '{"entities": [{"name": "...", "type": "...", "description": "...", "aliases": []}], '
    '"triples": [{"subject": "...", "predicate": "snake_case_verb", "object": "...", '
    '"description": "...", "negated": false, "valid_from": null, "valid_to": null}]}.'
)

_CHUNKS_AND_FRONTMATTER_NOTE = (
    "Indexed document chunks and YAML frontmatter triples without an external chat LLM. "
    "For prose triple extraction in Zero-Key Mode, run the 'sync-adr-folder' prompt "
    "(calling 'remember' with structured triples per ADR) or set OPENAI_API_KEY."
)


def _normalize_yaml_values(value: Any) -> Any:
    """Recursively convert date/datetime instances produced by PyYAML into ISO strings."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize_yaml_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml_values(item) for item in value]
    return value


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (idx == 0 or line[idx - 1] != "\\"):
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx].rstrip()
    return line.rstrip()


def _split_inline_items(body: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    for idx, ch in enumerate(body):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (idx == 0 or body[idx - 1] != "\\"):
            in_double = not in_double
        elif not in_single and not in_double:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    items.append(part)
                buf = []
                continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def _parse_scalar_or_inline(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return None
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar_or_inline(part) for part in _split_inline_items(inner)]
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            if not inner:
                return {}
            out_map: dict[str, Any] = {}
            for pair in _split_inline_items(inner):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    out_map[_unquote(k.strip())] = _parse_scalar_or_inline(v)
            return out_map

    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return _unquote(text)

    low = text.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _unquote(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        inner = val[1:-1]
        if val[0] == '"':
            return inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner.replace("''", "'")
    return val


def _split_key_value(text: str) -> tuple[str, str] | None:
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single and (idx == 0 or text[idx - 1] != "\\"):
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            if idx + 1 == len(text) or text[idx + 1].isspace():
                return (_unquote(text[:idx].strip()), text[idx + 1 :].strip())
    return None


def _parse_yaml_block(lines: list[tuple[int, str]], start: int, min_indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return (None, start)
    first_indent, first_content = lines[start]
    if first_indent < min_indent:
        return (None, start)

    if first_content.startswith("- ") or first_content == "-":
        items: list[Any] = []
        idx = start
        list_indent = first_indent
        while idx < len(lines):
            indent, content = lines[idx]
            if indent < list_indent:
                break
            if indent != list_indent or not (content.startswith("- ") or content == "-"):
                break
            after_dash = content[1:].strip()
            idx += 1
            if not after_dash:
                nested_val, idx = _parse_yaml_block(lines, idx, list_indent + 1)
                items.append(nested_val)
                continue

            kv = _split_key_value(after_dash)
            if kv is not None and not after_dash.startswith("{"):
                k, v_raw = kv
                item_dict: dict[str, Any] = {}
                if v_raw:
                    item_dict[k] = _parse_scalar_or_inline(v_raw)
                else:
                    nested_val, idx = _parse_yaml_block(lines, idx, list_indent + 2)
                    item_dict[k] = nested_val

                while idx < len(lines) and lines[idx][0] > list_indent:
                    sub_indent, sub_content = lines[idx]
                    sub_kv = _split_key_value(sub_content)
                    if sub_kv is None:
                        idx += 1
                        continue
                    sk, sv_raw = sub_kv
                    idx += 1
                    if sv_raw:
                        item_dict[sk] = _parse_scalar_or_inline(sv_raw)
                    else:
                        nested_val, idx = _parse_yaml_block(lines, idx, sub_indent + 1)
                        item_dict[sk] = nested_val
                items.append(item_dict)
            else:
                items.append(_parse_scalar_or_inline(after_dash))
        return (items, idx)

    mapping: dict[str, Any] = {}
    idx = start
    map_indent = first_indent
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < map_indent:
            break
        if indent > map_indent:
            idx += 1
            continue
        kv = _split_key_value(content)
        if kv is None:
            idx += 1
            continue
        key, val_raw = kv
        idx += 1
        if val_raw:
            mapping[key] = _parse_scalar_or_inline(val_raw)
        else:
            nested_val, idx = _parse_yaml_block(lines, idx, map_indent + 1)
            mapping[key] = nested_val
    return (mapping, idx)


def _parse_yaml_subset(frontmatter_raw: str) -> dict[str, Any]:
    """Parse YAML frontmatter without external dependencies when PyYAML is absent."""
    stripped = frontmatter_raw.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}

    prepared: list[tuple[int, str]] = []
    for raw_line in frontmatter_raw.splitlines():
        cleaned = _strip_yaml_comment(raw_line)
        if not cleaned.strip():
            continue
        expanded = cleaned.expandtabs(2)
        indent = len(expanded) - len(expanded.lstrip(" "))
        prepared.append((indent, expanded.strip()))

    if not prepared:
        return {}
    result, _ = _parse_yaml_block(prepared, 0, 0)
    return result if isinstance(result, dict) else {}


def _load_frontmatter_dict(frontmatter_raw: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return _parse_yaml_subset(frontmatter_raw)

    loaded = yaml.safe_load(frontmatter_raw)
    normalized = _normalize_yaml_values(loaded)
    return normalized if isinstance(normalized, dict) else {}


def parse_markdown_frontmatter(
    raw_text: str,
) -> tuple[dict[str, Any], ExtractionResult, list[str], str, str]:
    """Extract optional YAML frontmatter metadata, entities/triples, and Markdown body.

    Returns ``(meta_dict, extraction_result, functional_predicates, body_text, fm_hash)``.
    """
    match = _FRONTMATTER_RE.match(raw_text)
    if match is None:
        return ({}, ExtractionResult(entities=[], triples=[]), [], raw_text, "")

    fm_raw = match.group(1)
    body_text = raw_text[match.end() :]
    fm_hash = hashlib.sha256(fm_raw.strip().encode("utf-8")).hexdigest()[:16]
    data = _load_frontmatter_dict(fm_raw)

    entities: list[Entity] = []
    for raw_ent in data.get("entities") or []:
        if isinstance(raw_ent, dict) and raw_ent.get("name"):
            aliases_raw = raw_ent.get("aliases") or []
            aliases = (
                [str(a) for a in aliases_raw if a is not None]
                if isinstance(aliases_raw, list)
                else [str(aliases_raw)]
            )
            entities.append(
                Entity(
                    name=str(raw_ent["name"]),
                    type=str(raw_ent.get("type") or "Concept"),
                    description=str(raw_ent.get("description") or ""),
                    aliases=aliases,
                )
            )
        elif isinstance(raw_ent, str) and raw_ent.strip():
            entities.append(Entity(name=raw_ent.strip(), type="Concept", description=""))

    triples: list[Triple] = []
    functional_preds: list[str] = []
    for extra_fp in data.get("functional_predicates") or []:
        if extra_fp:
            functional_preds.append(str(extra_fp))

    default_valid_from = (
        str(data["valid_from"]) if data.get("valid_from") is not None else None
    )
    default_valid_to = str(data["valid_to"]) if data.get("valid_to") is not None else None

    for raw_triple in data.get("triples") or []:
        if isinstance(raw_triple, dict):
            subj = raw_triple.get("subject")
            pred = raw_triple.get("predicate")
            obj = raw_triple.get("object")
            if subj is None or pred is None or obj is None:
                continue
            if bool(raw_triple.get("functional", False)):
                functional_preds.append(str(pred))
            vf = raw_triple.get("valid_from", default_valid_from)
            vt = raw_triple.get("valid_to", default_valid_to)
            triples.append(
                Triple(
                    subject=str(subj),
                    predicate=str(pred),
                    object=str(obj),
                    description=(
                        str(raw_triple["description"])
                        if raw_triple.get("description") is not None
                        else None
                    ),
                    negated=bool(raw_triple.get("negated", False)),
                    confidence=float(raw_triple.get("confidence", 1.0)),
                    valid_from=str(vf) if vf is not None else None,
                    valid_to=str(vt) if vt is not None else None,
                )
            )
        elif isinstance(raw_triple, list) and len(raw_triple) >= 3:
            triples.append(
                Triple(
                    subject=str(raw_triple[0]),
                    predicate=str(raw_triple[1]),
                    object=str(raw_triple[2]),
                    valid_from=default_valid_from,
                    valid_to=default_valid_to,
                )
            )

    scalar_meta = {
        k: v
        for k, v in data.items()
        if k not in ("entities", "triples", "functional_predicates")
    }
    return (
        scalar_meta,
        ExtractionResult(entities=entities, triples=triples),
        functional_preds,
        body_text,
        fm_hash,
    )


def _relative_source_label(file_path: Path, allowed_roots: list[Path]) -> str:
    matching_roots = [
        root for root in allowed_roots if file_path == root or file_path.is_relative_to(root)
    ]
    if matching_roots:
        best_root = max(matching_roots, key=lambda r: len(r.parts))
        if file_path != best_root:
            return file_path.relative_to(best_root).as_posix()
    return file_path.name


def _git_first_commit_timestamp(file_path: Path) -> int | None:
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--follow",
                "--format=%ct",
                "-1",
                "--",
                str(file_path),
            ],
            cwd=str(file_path.parent),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    first_line = out.splitlines()[0].strip()
    return int(first_line) if first_line.lstrip("-").isdigit() else None


def _discover_and_sort_files_sync(
    mcp_cfg: MCPServerConfig,
    paths: list[str],
    glob_pattern: str,
    order_by: OrderBy,
) -> list[tuple[Path, str]]:
    if not paths:
        raise ValueError("Parameter 'paths' must contain at least one file or directory path.")
    if order_by not in ("filename", "mtime", "git_first_commit"):
        raise ValueError(
            f"Invalid order_by={order_by!r}; expected 'filename', 'mtime', or 'git_first_commit'."
        )

    seen: set[Path] = set()
    discovered: list[tuple[Path, str]] = []

    for raw_path in paths:
        resolved = mcp_cfg.validate_ingest_path(raw_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Ingest path does not exist: '{raw_path}'.")

        if resolved.is_file():
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(
                    (resolved, _relative_source_label(resolved, mcp_cfg.allowed_roots))
                )
        elif resolved.is_dir():
            for candidate in sorted(resolved.glob(glob_pattern)):
                if not candidate.is_file():
                    continue
                validated_child = mcp_cfg.validate_ingest_path(candidate)
                if validated_child not in seen:
                    seen.add(validated_child)
                    discovered.append(
                        (
                            validated_child,
                            _relative_source_label(validated_child, mcp_cfg.allowed_roots),
                        )
                    )

    if order_by == "filename":
        discovered.sort(key=lambda item: (item[1], str(item[0])))
    elif order_by == "mtime":
        discovered.sort(key=lambda item: (item[0].stat().st_mtime, item[1]))
    elif order_by == "git_first_commit":
        keyed: list[tuple[tuple[int, int, str], tuple[Path, str]]] = []
        for path_obj, rel_label in discovered:
            ts = _git_first_commit_timestamp(path_obj)
            sort_key = (0, ts, rel_label) if ts is not None else (1, 0, rel_label)
            keyed.append((sort_key, (path_obj, rel_label)))
        keyed.sort(key=lambda entry: entry[0])
        discovered = [entry[1] for entry in keyed]

    return discovered


def _client_supports_sampling(ctx: Context | None) -> Any:
    """Return the active MCP session if client-side sampling is supported, else None."""
    if ctx is None:
        return None
    try:
        session = ctx.session
    except Exception:
        return None
    if session is None or not hasattr(session, "create_message"):
        return None
    if hasattr(session, "check_client_capability"):
        try:
            supported = session.check_client_capability(
                ClientCapabilities(sampling=SamplingCapability())
            )
            return session if supported else None
        except Exception:
            return None
    return session


async def _sample_extraction_via_client(
    session: Any,
    chunk_text: str,
    doc_context: DocumentContext,
) -> ExtractionResult | None:
    ctx_prefix = f"Document: {doc_context.title or doc_context.source or 'document'}\n"
    if doc_context.known_entities:
        ctx_prefix += f"Known entities: {', '.join(doc_context.known_entities[:25])}\n"

    result = await session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"{ctx_prefix}\n---\n{chunk_text}",
                ),
            )
        ],
        max_tokens=2048,
        system_prompt=_SAMPLING_SYSTEM_PROMPT,
        temperature=0.0,
    )
    content = getattr(result, "content", None)
    raw_json = getattr(content, "text", None) if content is not None else None
    if not isinstance(raw_json, str) or not raw_json.strip():
        return ExtractionResult(entities=[], triples=[])
    parsed = GraphExtractor._parse_json_result(raw_json)
    return parsed if parsed is not None else ExtractionResult(entities=[], triples=[])


async def run_ingest_documents(
    *,
    mcp_cfg: MCPServerConfig,
    manager: EngineManager,
    paths: list[str],
    glob: str = "**/*.md",
    order_by: OrderBy = "filename",
    collection: str | None = None,
    space: str | None = None,
    build_communities: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Validate, deterministically sort, and ingest Markdown/ADR files into :class:`GraphRAG`."""
    if build_communities and not mcp_cfg.autonomous_llm_enabled:
        raise ValueError(
            "build_communities=True requires an autonomous chat LLM (set OPENAI_API_KEY) "
            "because community summarization invokes chat completions."
        )

    files = await asyncio.to_thread(
        _discover_and_sort_files_sync, mcp_cfg, paths, glob, order_by
    )

    rag: GraphRAG = await manager.get_rag()
    target_space = space or mcp_cfg.default_space

    sampling_session = (
        None if mcp_cfg.autonomous_llm_enabled else _client_supports_sampling(ctx)
    )
    if mcp_cfg.autonomous_llm_enabled:
        extraction_mode: ExtractionMode = "llm_graph"
    elif sampling_session is not None:
        extraction_mode = "mcp_sampling"
    else:
        extraction_mode = "chunks_and_frontmatter"

    files_ingested = 0
    files_skipped_unchanged = 0
    chunks_indexed = 0
    chunks_skipped_unchanged = 0
    entities_extracted = 0
    triples_extracted = 0
    relations_added = 0
    superseded_ids_all: list[str] = []
    superseded_details_all: list[dict[str, Any]] = []

    empty_extraction = ExtractionResult(entities=[], triples=[])

    for file_path, rel_source in files:
        raw_content = await asyncio.to_thread(file_path.read_text, "utf-8")
        fm_meta, fm_extraction, fm_functional, body_text, fm_hash = parse_markdown_frontmatter(
            raw_content
        )

        if fm_functional:
            manager.extend_rules(add_functional=fm_functional)

        index_text_body = body_text.strip() or raw_content.strip()
        if not index_text_body and not fm_extraction.entities and not fm_extraction.triples:
            continue

        chunks = rag.chunker(index_text_body) if index_text_body else [""]
        if not chunks:
            chunks = [index_text_body]

        doc_title = str(fm_meta.get("title") or fm_meta.get("document") or file_path.name)
        doc_category = str(fm_meta["category"]) if fm_meta.get("category") else "document"
        doc_key_val = document_key(rel_source, doc_title)

        existing = await rag.store.find_document_chunks(doc_key_val, space=target_space)
        incoming_hashes = [content_hash(c) for c in chunks]
        existing_fm_hash = (
            str((existing[0].get("payload") or {}).get("frontmatter_hash") or "")
            if existing
            else ""
        )
        is_unchanged = (
            bool(existing)
            and [e.get("content_hash") for e in existing] == incoming_hashes
            and existing_fm_hash == fm_hash
        )

        if is_unchanged and rag.config.skip_unchanged_documents:
            files_skipped_unchanged += 1
            chunks_skipped_unchanged += len(existing)
            continue

        if existing:
            await rag.store.delete_document_chunks(doc_key_val, space=target_space)

        extra_payload: dict[str, Any] = {"frontmatter_hash": fm_hash}
        for mk, mv in fm_meta.items():
            if mk not in ("title", "document", "category", "source", "collection", "space"):
                extra_payload[mk] = mv

        doc_ctx = DocumentContext(title=doc_title, source=rel_source, known_entities=[])
        has_fm_graph = bool(fm_extraction.entities or fm_extraction.triples)

        for idx, chunk_str in enumerate(chunks):
            chunk_meta = DocumentMetadata(
                source=rel_source,
                document=doc_title,
                category=doc_category,
                collection=collection,
                paragraph=idx + 1,
                space=target_space,
                extra=dict(extra_payload),
            )

            chunk_ext: ExtractionResult | None
            if extraction_mode == "llm_graph":
                if has_fm_graph and idx == 0:
                    prose_ext = await rag._extract_or_empty(chunk_str, doc_ctx)
                    chunk_ext = GraphExtractor._merge(fm_extraction, prose_ext)
                else:
                    chunk_ext = None
            elif extraction_mode == "mcp_sampling" and sampling_session is not None:
                try:
                    sampled = await _sample_extraction_via_client(
                        sampling_session, chunk_str, doc_ctx
                    )
                    base_sampled = sampled or empty_extraction
                    chunk_ext = (
                        GraphExtractor._merge(fm_extraction, base_sampled)
                        if (has_fm_graph and idx == 0)
                        else base_sampled
                    )
                except Exception as exc:
                    logger.warning(
                        "MCP client sampling unavailable during ingest (%s); "
                        "falling back to chunks_and_frontmatter mode.",
                        exc,
                    )
                    sampling_session = None
                    extraction_mode = "chunks_and_frontmatter"
                    chunk_ext = fm_extraction if (has_fm_graph and idx == 0) else empty_extraction
            else:
                chunk_ext = fm_extraction if (has_fm_graph and idx == 0) else empty_extraction

            write_res = await rag.index_document(
                chunk_str,
                metadata=chunk_meta,
                space=target_space,
                context=doc_ctx,
                extraction=chunk_ext,
            )

            chunks_indexed += 1
            entities_extracted += int(write_res.get("entities_extracted", 0))
            triples_extracted += int(write_res.get("triples_extracted", 0))
            relations_added += int(write_res.get("relations_added", 0))

            for sid in write_res.get("superseded_ids") or []:
                s_str = str(sid)
                if s_str not in superseded_ids_all:
                    superseded_ids_all.append(s_str)
            for detail in write_res.get("superseded_details") or []:
                superseded_details_all.append(detail)

            for ent_name in write_res.get("entities") or []:
                if ent_name not in doc_ctx.known_entities:
                    doc_ctx.known_entities.append(ent_name)

        files_ingested += 1

    communities_result: dict[str, Any] | None = None
    if build_communities:
        communities_result = await rag.build_communities(space=target_space)

    wm = await rag.watermark(space=target_space)
    response: dict[str, Any] = {
        "files_scanned": len(files),
        "files_ingested": files_ingested,
        "files_skipped_unchanged": files_skipped_unchanged,
        "chunks_indexed": chunks_indexed,
        "chunks_skipped_unchanged": chunks_skipped_unchanged,
        "entities_extracted": entities_extracted,
        "triples_extracted": triples_extracted,
        "relations_added": relations_added,
        "superseded_count": max(len(superseded_ids_all), len(superseded_details_all)),
        "superseded_relations": superseded_details_all,
        "extraction_mode": extraction_mode,
        "communities_built": communities_result,
        "watermark": wm,
    }
    if extraction_mode == "chunks_and_frontmatter":
        response["note"] = _CHUNKS_AND_FRONTMATTER_NOTE

    return response
