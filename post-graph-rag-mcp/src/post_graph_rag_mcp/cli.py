"""Command-line entrypoint for post-graph-rag-mcp (serve, doctor, and init)."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib.util
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from post_graph_rag import RAGConfig

from post_graph_rag_mcp import __version__
from post_graph_rag_mcp.config import MCPServerConfig
from post_graph_rag_mcp.embeddings import build_llm_service
from post_graph_rag_mcp.engine_manager import EngineManager
from post_graph_rag_mcp.server import create_mcp_server

_T = TypeVar("_T")


def _run_async_entry(coro_factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run an async CLI routine synchronously, even if invoked inside an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


def _default_transport() -> str:
    val = (os.getenv("MCP_TRANSPORT") or "stdio").strip().lower()
    return val if val in ("stdio", "streamable-http") else "stdio"


def _default_host() -> str:
    return (os.getenv("MCP_HOST") or "127.0.0.1").strip() or "127.0.0.1"


def _default_port() -> int:
    raw = (os.getenv("MCP_PORT") or "8080").strip()
    return int(raw) if raw.isdigit() else 8080


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=_default_transport(),
        help="MCP wire transport (default: MCP_TRANSPORT or 'stdio').",
    )
    parser.add_argument(
        "--host",
        default=_default_host(),
        help="Bind host for streamable-http transport (default: MCP_HOST or '127.0.0.1').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help="Bind port for streamable-http transport (default: MCP_PORT or 8080).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser supporting ``serve`` (default), ``doctor``, and ``init``."""
    parser = argparse.ArgumentParser(
        prog="post-graph-rag-mcp",
        description="Bi-temporal GraphRAG memory MCP server backed by PostgreSQL + pgvector.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"post-graph-rag-mcp {__version__}",
    )
    _add_serve_arguments(parser)

    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the MCP server over stdio or streamable-http (default action).",
    )
    _add_serve_arguments(serve_parser)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run a live end-to-end PostgreSQL + pgvector + embedding health check.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_only",
        help="Emit machine-readable JSON only.",
    )

    subparsers.add_parser(
        "init",
        help="Pre-warm embedded PostgreSQL / local ONNX weights and print IDE config snippets.",
    )

    return parser


async def _run_doctor_async() -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    cfg = MCPServerConfig.from_env()
    manager = EngineManager(cfg)
    try:
        rag = await manager.get_rag()
        target_space = cfg.default_space
        watermark = await rag.watermark(space=target_space)
        delta = await rag.changes_since(
            "1970-01-01T00:00:00+00:00",
            space=target_space,
            include=("documents", "entities", "relations"),
            summary=False,
            limit=100,
        )
        doc_keys = [str(d["doc_key"]) for d in delta.new_documents if d.get("doc_key")]
        doc_stats = await rag.documents_stats(doc_keys, space=target_space)

        probe_vec = await rag.llm.get_embedding("post-graph-rag-mcp doctor probe")
        probe_dim = len(probe_vec)
        if probe_dim != cfg.embedding_dim:
            raise RuntimeError(
                f"Embedding probe returned {probe_dim} dimensions, "
                f"expected {cfg.embedding_dim} for '{cfg.embedding_model}'."
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        report: dict[str, Any] = {
            "status": "healthy",
            "version": __version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "realm": cfg.realm,
            "space": target_space,
            "preset": manager.preset_name,
            "storage_mode": "external_postgres" if cfg.postgres_uri else "embedded_pgserver",
            "embedding_backend": cfg.embedding_backend,
            "embedding_model": cfg.embedding_model,
            "embedding_dim": cfg.embedding_dim,
            "embedding_probe_dim": probe_dim,
            "autonomous_llm_enabled": cfg.autonomous_llm_enabled,
            "watermark": watermark,
            "documents_indexed": delta.counts.get("new_documents", 0),
            "documents_stats_checked": len(doc_stats),
            "counts": dict(delta.counts),
            "elapsed_ms": elapsed_ms,
        }
        return (0, report)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        error_report: dict[str, Any] = {
            "status": "unhealthy",
            "version": __version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "realm": cfg.realm,
            "storage_mode": "external_postgres" if cfg.postgres_uri else "embedded_pgserver",
            "embedding_backend": cfg.embedding_backend,
            "embedding_model": cfg.embedding_model,
            "embedding_dim": cfg.embedding_dim,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_ms": elapsed_ms,
        }
        return (1, error_report)
    finally:
        try:
            await manager.close()
        except Exception:
            pass


def _run_doctor(json_only: bool = False) -> int:
    code, report = _run_async_entry(_run_doctor_async)
    stream = sys.stdout if code == 0 else sys.stderr
    if not json_only:
        headline = (
            f"[post-graph-rag-mcp doctor] HEALTHY ({report['elapsed_ms']} ms)"
            if code == 0
            else f"[post-graph-rag-mcp doctor] UNHEALTHY: {report.get('error_type')}: {report.get('error')}"
        )
        print(headline, file=stream)
    print(json.dumps(report, indent=2), file=sys.stdout if json_only else stream)
    return code


async def _run_init_async() -> dict[str, Any]:
    cfg = MCPServerConfig.from_env()
    steps: dict[str, Any] = {
        "realm": cfg.realm,
        "pgdata_dir": str(cfg.pgdata_dir),
        "postgres_warmed": False,
        "embeddings_warmed": False,
    }

    has_pgserver = importlib.util.find_spec("pgserver") is not None and sys.version_info < (3, 13)
    has_fastembed = importlib.util.find_spec("fastembed") is not None

    if cfg.embedding_backend == "fastembed" and has_fastembed:
        rag_cfg = RAGConfig(
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
        )
        llm_service = build_llm_service(cfg, rag_cfg)
        vec = await llm_service.get_embedding("post-graph-rag-mcp warmup")
        steps["embeddings_warmed"] = len(vec) == cfg.embedding_dim
        steps["embedding_model"] = cfg.embedding_model
        steps["embedding_dim"] = len(vec)

    if cfg.postgres_uri or has_pgserver:
        manager = EngineManager(cfg)
        try:
            rag = await manager.get_rag()
            steps["watermark"] = await rag.watermark(space=cfg.default_space)
            steps["postgres_warmed"] = True
        finally:
            await manager.close()
    else:
        steps["postgres_note"] = (
            "POSTGRES_URI is unset and 'pgserver' is not installed in this interpreter. "
            "Use --python 3.12 with [embedded,local-embed] for zero-config embedded PostgreSQL, "
            "or supply POSTGRES_URI for any PostgreSQL 16-18 + pgvector instance."
        )

    return steps


def _print_ide_snippets(cfg: MCPServerConfig) -> None:
    tier1_cursor = {
        "mcpServers": {
            "post-graph-rag": {
                "command": "uvx",
                "args": [
                    "--python",
                    "3.12",
                    "--from",
                    "post-graph-rag-mcp[embedded,local-embed]",
                    "post-graph-rag-mcp",
                ],
            }
        }
    }
    tier2_uri = cfg.postgres_uri or "postgresql://localhost:5432/postgres"
    tier2_cursor = {
        "mcpServers": {
            "post-graph-rag": {
                "command": "uvx",
                "args": [
                    "--from",
                    "post-graph-rag-mcp[local-embed]",
                    "post-graph-rag-mcp",
                ],
                "env": {
                    "POSTGRES_URI": tier2_uri,
                    "RAG_REALM": cfg.realm,
                },
            }
        }
    }

    print("\n=== Tier 1: Zero-Config Embedded PostgreSQL + Local ONNX (Python 3.12) ===")
    print("Claude Code:")
    print(
        '  claude mcp add post-graph-rag -- uvx --python 3.12 '
        '--from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp'
    )
    print("\nCursor (.cursor/mcp.json):")
    print(json.dumps(tier1_cursor, indent=2))

    print("\n=== Tier 2: Existing PostgreSQL / Neon / Supabase (Python 3.11-3.14) ===")
    print("Claude Code:")
    print(
        f'  claude mcp add post-graph-rag -e POSTGRES_URI="{tier2_uri}" '
        f'-e RAG_REALM="{cfg.realm}" '
        '-- uvx --from "post-graph-rag-mcp[local-embed]" post-graph-rag-mcp'
    )
    print("\nCursor (.cursor/mcp.json):")
    print(json.dumps(tier2_cursor, indent=2))


def _run_init() -> int:
    cfg = MCPServerConfig.from_env()
    summary = _run_async_entry(_run_init_async)
    print("[post-graph-rag-mcp init] Pre-warm summary:")
    print(json.dumps(summary, indent=2))
    _print_ide_snippets(cfg)
    return 0


def _run_serve(transport: str, host: str, port: int) -> int:
    logging.basicConfig(
        level=getattr(logging, (os.getenv("MCP_LOG_LEVEL") or "WARNING").upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    cfg = MCPServerConfig.from_env()
    server = create_mcp_server(mcp_cfg=cfg, host=host, port=port)
    server.run(transport=transport)  # type: ignore[arg-type]
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning process exit status (0 on success, 1 on failure)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _run_doctor(json_only=bool(args.json_only))
    if args.command == "init":
        return _run_init()
    return _run_serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
