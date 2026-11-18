"""Live-PostgreSQL MCP JSON-RPC protocol roundtrip tests (in-memory streams, stdio subprocess, and CLI)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_client_server_memory_streams

from post_graph_rag_mcp.cli import main as cli_main
from post_graph_rag_mcp.config import MCPServerConfig
from post_graph_rag_mcp.engine_manager import EngineManager
from post_graph_rag_mcp.server import create_mcp_server

EXPECTED_TOOLS = {
    "remember",
    "recall",
    "recall_as_of",
    "trace_history",
    "ingest_documents",
    "changes_since",
    "configure_memory_rules",
}

EXPECTED_RESOURCES = {
    "memory://status",
    "memory://rules",
    "memory://communities",
}

EXPECTED_PROMPTS = {
    "remember-architecture-decision",
    "audit-belief-state",
    "sync-adr-folder",
}


def _parse_tool_result(call_result: Any) -> dict[str, Any]:
    assert call_result.isError is False, f"MCP tool returned error: {call_result.content}"
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    assert call_result.content, "Expected non-empty content list from call_tool"
    text = call_result.content[0].text
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


async def test_fast_initialize_handshake_before_engine_warmup(
    clean_mcp_env: Path,
) -> None:
    t_construct = time.perf_counter()
    cfg = MCPServerConfig.from_env(cwd=clean_mcp_env)
    mgr = EngineManager(cfg)
    srv = create_mcp_server(mcp_cfg=cfg, manager=mgr)
    construct_ms = (time.perf_counter() - t_construct) * 1000.0
    assert construct_ms < 100.0, f"Server construction took {construct_ms:.2f}ms"

    try:
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            client_read, client_write = client_streams
            server_read, server_write = server_streams
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    srv._mcp_server.run,
                    server_read,
                    server_write,
                    srv._mcp_server.create_initialization_options(),
                )
                async with ClientSession(client_read, client_write) as session:
                    t0 = time.perf_counter()
                    init_res = await session.initialize()
                    handshake_ms = (time.perf_counter() - t0) * 1000.0

                    assert handshake_ms < 100.0, (
                        f"MCP initialize handshake took {handshake_ms:.2f}ms (>= 100ms)"
                    )
                    assert init_res.instructions is not None
                    assert len(init_res.instructions.strip()) > 20
                    assert "recall" in init_res.instructions
                    assert "remember" in init_res.instructions
                tg.cancel_scope.cancel()
    finally:
        await mgr.close()


async def test_full_jsonrpc_client_session_roundtrip_against_live_postgres(
    mcp_harness: Any,
) -> None:
    srv = mcp_harness.server

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                srv._mcp_server.run,
                server_read,
                server_write,
                srv._mcp_server.create_initialization_options(),
            )
            async with ClientSession(client_read, client_write) as session:
                init_res = await session.initialize()
                assert init_res.instructions and init_res.instructions.strip()

                tools_res = await session.list_tools()
                tool_names = {t.name for t in tools_res.tools}
                assert len(tools_res.tools) == 7
                assert tool_names == EXPECTED_TOOLS
                for tool in tools_res.tools:
                    assert tool.description and len(tool.description.strip()) > 10

                resources_res = await session.list_resources()
                resource_uris = {str(r.uri) for r in resources_res.resources}
                assert len(resources_res.resources) == 3
                assert resource_uris == EXPECTED_RESOURCES

                prompts_res = await session.list_prompts()
                prompt_names = {p.name for p in prompts_res.prompts}
                assert len(prompts_res.prompts) == 3
                assert prompt_names == EXPECTED_PROMPTS

                for prompt in prompts_res.prompts:
                    sample_args = {
                        arg.name: "adr"
                        for arg in (prompt.arguments or [])
                        if arg.required
                    }
                    rendered = await session.get_prompt(prompt.name, sample_args)
                    assert len(rendered.messages) >= 1

                rem_raw = await session.call_tool(
                    "remember",
                    {
                        "text": "OrderService persists transactions in PostgreSQL.",
                        "triples": [
                            {
                                "subject": "OrderService",
                                "predicate": "uses_database",
                                "object": "PostgreSQL",
                            }
                        ],
                    },
                )
                rem_data = _parse_tool_result(rem_raw)
                assert rem_data["relations_written"] == 1

                rec_raw = await session.call_tool(
                    "recall",
                    {
                        "question": "Which database does OrderService use?",
                        "entity_keywords": ["OrderService"],
                    },
                )
                rec_data = _parse_tool_result(rec_raw)
                assert any(
                    mcp_harness.subject(r) == "OrderService"
                    and mcp_harness.object(r) == "PostgreSQL"
                    for r in rec_data["relationships"]
                )

                status_res = await session.read_resource("memory://status")
                assert status_res.contents
                status_payload = json.loads(status_res.contents[0].text)
                assert status_payload["realm"] == mcp_harness.config.realm
                assert status_payload["watermark"]

                rules_res = await session.read_resource("memory://rules")
                assert rules_res.contents
                rules_payload = json.loads(rules_res.contents[0].text)
                rules_body = rules_payload.get("rules", rules_payload)
                assert "uses_database" in set(rules_body["functional_predicates"])

                comm_res = await session.read_resource("memory://communities")
                assert comm_res.contents
                comm_payload = json.loads(comm_res.contents[0].text)
                comm_tree = (
                    comm_payload.get("tree")
                    or comm_payload.get("communities")
                    or comm_payload
                )
                assert "levels" in comm_tree

            tg.cancel_scope.cancel()


async def test_stdio_subprocess_jsonrpc_handshake_and_inventory(
    mcp_harness_factory: Callable[..., Any],
    clean_mcp_env: Path,
) -> None:
    harness = mcp_harness_factory()
    await harness.manager.get_rag()

    pythonpath = os.pathsep.join(p for p in sys.path if p)
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "POSTGRES_URI": harness.postgres_uri,
        "RAG_REALM": harness.config.realm,
        "MCP_ALLOWED_ROOTS": str(clean_mcp_env),
    }
    env.pop("OPENAI_API_KEY", None)
    env.pop("RAG_API_KEY", None)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from post_graph_rag_mcp.cli import main; main()"],
        env=env,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init_res = await session.initialize()
            assert init_res.instructions and "recall" in init_res.instructions

            tools_res = await session.list_tools()
            assert {t.name for t in tools_res.tools} == EXPECTED_TOOLS

            resources_res = await session.list_resources()
            assert {str(r.uri) for r in resources_res.resources} == EXPECTED_RESOURCES

            prompts_res = await session.list_prompts()
            assert {p.name for p in prompts_res.prompts} == EXPECTED_PROMPTS

            status_res = await session.read_resource("memory://status")
            status_data = json.loads(status_res.contents[0].text)
            assert status_data["realm"] == harness.config.realm


@pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is None,
    reason="doctor CLI end-to-end embedding probe requires fastembed in Mode A",
)
async def test_cli_doctor_reports_healthy_against_live_postgres(
    mcp_harness_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = mcp_harness_factory(use_real_fastembed=True)
    await harness.manager.get_rag()
    monkeypatch.setenv("POSTGRES_URI", harness.postgres_uri)
    monkeypatch.setenv("RAG_REALM", harness.config.realm)

    try:
        rc = await asyncio.to_thread(cli_main, ["doctor"])
    except SystemExit as exc:
        rc = exc.code if exc.code is not None else 0

    assert rc == 0
    captured = capsys.readouterr()
    combined_out = captured.out + captured.err
    assert harness.config.realm in combined_out
