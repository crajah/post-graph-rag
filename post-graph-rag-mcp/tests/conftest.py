"""Shared live-PostgreSQL fixtures for post-graph-rag-mcp tests.

Every test runs against a real PostgreSQL + pgvector instance via MCPServerConfig,
EngineManager, and create_mcp_server, isolated in its own per-test schema realm
(schema_per_realm=True) and cleaned up via AsyncPostGraph.drop_realm() on teardown.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import re
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from post_graph_rag import GraphRAG, RAGConfig
from post_graph_rag.errors import EmbeddingError, LLMError
from post_graph_rag.llm import LLMService

TEST_POSTGRES_URI = os.getenv(
    "POSTGRES_TEST_URI",
    os.getenv("POSTGRES_URI", "postgresql://localhost:5432/postgres"),
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _postgres_socket_reachable(uri: str) -> bool:
    """Verify TCP or Unix-domain socket reachability without any SQL driver."""
    parsed = urlparse(uri)
    qs = parse_qs(parsed.query)
    host = (qs.get("host", [None])[0]) or parsed.hostname or "localhost"
    port = int((qs.get("port", [None])[0]) or parsed.port or 5432)

    if host.startswith("/"):
        sock_path = Path(host) / f".s.PGSQL.{port}"
        return sock_path.exists()

    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def deterministic_unit_vector(text: str, dim: int) -> list[float]:
    """Deterministic SHA-256 token + character-trigram unit vector of width *dim*.

    Preserves lexical similarity in pgvector HNSW cosine space without network
    or model-weight downloads, while guaranteeing finite unit-norm vectors for
    every non-empty or empty input string.
    """
    if dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {dim}")

    tokens = _TOKEN_RE.findall((text or "").lower())
    features: list[tuple[str, float]] = [(tok, 1.0) for tok in tokens]
    for tok in tokens:
        if len(tok) >= 3:
            for idx in range(len(tok) - 2):
                features.append((f"3g:{tok[idx:idx + 3]}", 0.35))

    vec = [0.0] * dim
    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        bucket_seed = int.from_bytes(digest[:8], "big")
        for offset in range(8):
            dim_idx = (bucket_seed + offset) % dim
            centered = (digest[8 + offset] - 128.0) / 128.0
            vec[dim_idx] += weight * centered

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        fill = 1.0 / math.sqrt(dim)
        return [fill] * dim
    return [v / norm for v in vec]


class DeterministicEmbedLLMService(LLMService):
    """Deterministic local embedding service for fast Mode-A integration tests.

    Subclasses LLMService so any signature drift in post-graph-rag surfaces
    immediately, computes real unit-norm vectors stored in pgvector, and raises
    LLMError on any chat-completion attempt to verify zero chat-LLM usage.
    """

    def __init__(self, config: RAGConfig, fail_embeddings: bool = False) -> None:
        self.config = config
        self.client = None
        self.served: collections.Counter[str] = collections.Counter()
        self._fail_embeddings = fail_embeddings
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.chat_calls: list[Any] = []

    async def get_embedding(self, text: str) -> list[float]:
        if self._fail_embeddings:
            raise EmbeddingError("Deterministic embedding failure triggered by test")
        self.embed_calls.append(text)
        return deterministic_unit_vector(text, self.config.embedding_dim)

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self._fail_embeddings:
            raise EmbeddingError("Deterministic batch embedding failure triggered by test")
        items = list(texts)
        if not items:
            return []
        self.batch_calls.append(items)
        self.embed_calls.extend(items)
        return [deterministic_unit_vector(t, self.config.embedding_dim) for t in items]

    async def chat_completion(
        self, messages: Any, response_format: Any = None, role: str | None = None
    ) -> Any:
        self.chat_calls.append((messages, response_format, role))
        raise LLMError(
            "Chat LLM is disabled in Zero-Key Host-Agent Mode. "
            "Unexpected chat_completion call occurred."
        )

    async def chat_completion_stream(
        self, messages: Any, role: str | None = None
    ) -> AsyncIterator[str]:
        self.chat_calls.append((messages, None, role))
        raise LLMError(
            "Chat LLM is disabled in Zero-Key Host-Agent Mode. "
            "Unexpected chat_completion_stream call occurred."
        )
        yield ""  # pragma: no cover


def new_test_realm() -> str:
    """Generate a unique per-test PostgreSQL schema realm identifier."""
    return f"mcp_test_{uuid.uuid4().hex[:10]}"


def update_mcp_config(cfg: Any, **overrides: Any) -> Any:
    """Return a copy of MCPServerConfig with *overrides* applied."""
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **overrides)
    return cfg.model_copy(update=overrides)


async def call_mcp_tool(
    server: Any, name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Invoke a FastMCP tool, unwrapping ToolError causes and JSON payloads."""
    try:
        raw = await server.call_tool(name, arguments or {})
    except Exception as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2:
        blocks, structured = raw
        if isinstance(structured, dict):
            return structured
        raw = blocks
    if isinstance(raw, list) and raw:
        first = raw[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
    raise TypeError(f"Unexpected return payload from tool {name!r}: {raw!r}")


def rel_subject(rel: dict[str, Any]) -> str:
    """Extract canonical subject name from a recall/trace relationship record."""
    return str(rel.get("subject") or rel.get("src_id") or "")


def rel_predicate(rel: dict[str, Any]) -> str:
    """Extract predicate / relation_type from a recall/trace relationship record."""
    return str(rel.get("predicate") or rel.get("relation_type") or "")


def rel_object(rel: dict[str, Any]) -> str:
    """Extract canonical object name from a recall/trace relationship record."""
    return str(rel.get("object") or rel.get("tgt_id") or "")


def rel_old_object(rel: dict[str, Any]) -> str:
    """Extract superseded object name from a supersession detail record."""
    return str(rel.get("old_object") or rel.get("object") or rel.get("tgt_id") or "")


def rel_id(rel: dict[str, Any]) -> str:
    """Extract edge identifier from a relationship record."""
    return str(rel.get("id") or rel.get("edge_id") or "")


def chunk_text(chunk: dict[str, Any]) -> str:
    """Extract text passage from a recall chunk record."""
    return str(chunk.get("text") or chunk.get("content") or "")


@dataclass
class MCPTestHarness:
    """Bound MCPServerConfig + EngineManager + FastMCP server for one test."""

    config: Any
    manager: Any
    server: Any
    llm_instances: list[DeterministicEmbedLLMService] = field(default_factory=list)
    opened_rags: list[GraphRAG] = field(default_factory=list)
    postgres_uri: str = TEST_POSTGRES_URI

    async def call(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await call_mcp_tool(self.server, name, arguments)

    @property
    def total_chat_calls(self) -> int:
        return sum(len(svc.chat_calls) for svc in self.llm_instances)

    @staticmethod
    def new_realm() -> str:
        return new_test_realm()

    @staticmethod
    def subject(rel: dict[str, Any]) -> str:
        return rel_subject(rel)

    @staticmethod
    def predicate(rel: dict[str, Any]) -> str:
        return rel_predicate(rel)

    @staticmethod
    def object(rel: dict[str, Any]) -> str:
        return rel_object(rel)

    @staticmethod
    def old_object(rel: dict[str, Any]) -> str:
        return rel_old_object(rel)

    @staticmethod
    def edge_id(rel: dict[str, Any]) -> str:
        return rel_id(rel)

    @staticmethod
    def chunk_text(chunk: dict[str, Any]) -> str:
        return chunk_text(chunk)


def track_manager_realms(manager: Any, opened_rags: list[GraphRAG]) -> Any:
    """Record every GraphRAG instance returned by manager.get_rag for realm cleanup."""
    original_get_rag = manager.get_rag

    async def _tracked_get_rag(realm: str | None = None) -> GraphRAG:
        rag = await original_get_rag(realm=realm)
        if all(existing is not rag for existing in opened_rags):
            opened_rags.append(rag)
        return rag

    manager.get_rag = _tracked_get_rag
    return manager


async def cleanup_managers(
    managers: list[Any], opened_rags: list[GraphRAG]
) -> None:
    """Drop all disposable per-test schemas via AsyncPostGraph.drop_realm, then close pools."""
    dropped_realms: set[str] = set()
    for rag in opened_rags:
        realm = rag.config.realm
        if realm in dropped_realms:
            continue
        try:
            await rag.store.client.drop_realm(realm)
            dropped_realms.add(realm)
        except Exception:
            pass

    for mgr in managers:
        try:
            await mgr.close()
        except Exception:
            pass


@pytest.fixture
def clean_mcp_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate environment variables so Mode-A zero-key behavior is deterministic."""
    for var in (
        "OPENAI_API_KEY",
        "RAG_API_KEY",
        "RAG_EMBEDDING_API_KEY",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIM",
        "RAG_REALM",
        "RAG_SPACE",
        "MCP_MEMORY_PRESET",
        "MCP_EMBEDDING_BACKEND",
        "MCP_FASTEMBED_MODEL",
        "MCP_ALLOWED_ROOTS",
        "RAG_CONTRADICTION_DETECTION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("POSTGRES_URI", TEST_POSTGRES_URI)
    monkeypatch.setenv("MCP_ALLOWED_ROOTS", str(tmp_path))
    return tmp_path


@pytest_asyncio.fixture
async def mcp_harness_factory(
    clean_mcp_env: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., MCPTestHarness]]:
    """Factory creating isolated live-PostgreSQL MCPServerConfig + EngineManager + FastMCP."""
    if not _postgres_socket_reachable(TEST_POSTGRES_URI):
        pytest.skip(f"PostgreSQL not reachable at {TEST_POSTGRES_URI}")

    import post_graph_rag_mcp.embeddings as mcp_embeddings
    import post_graph_rag_mcp.engine_manager as mcp_engine_manager
    from post_graph_rag_mcp.config import MCPServerConfig
    from post_graph_rag_mcp.engine_manager import EngineManager
    from post_graph_rag_mcp.server import create_mcp_server

    real_build_llm_service = mcp_embeddings.build_llm_service
    managers: list[Any] = []
    all_opened_rags: list[GraphRAG] = []

    def _make(
        *,
        use_real_fastembed: bool = False,
        realm: str | None = None,
        preset: str = "engineering",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        allowed_roots: list[Path] | None = None,
        **extra_overrides: Any,
    ) -> MCPTestHarness:
        target_realm = realm or new_test_realm()
        roots = [p.resolve() for p in (allowed_roots or [clean_mcp_env])]
        base_cfg = MCPServerConfig.from_env(cwd=clean_mcp_env)
        cfg = update_mcp_config(
            base_cfg,
            postgres_uri=TEST_POSTGRES_URI,
            realm=target_realm,
            preset=preset,
            embedding_backend="fastembed",
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            autonomous_llm_enabled=False,
            contradiction_detection=False,
            allowed_roots=roots,
            schema_per_realm=True,
            **extra_overrides,
        )

        llm_instances: list[DeterministicEmbedLLMService] = []
        if use_real_fastembed:
            monkeypatch.setattr(
                mcp_engine_manager, "build_llm_service", real_build_llm_service
            )
            monkeypatch.setattr(
                mcp_embeddings, "build_llm_service", real_build_llm_service
            )
        else:
            def _deterministic_builder(_mcp_cfg: Any, rag_cfg: RAGConfig) -> LLMService:
                svc = DeterministicEmbedLLMService(rag_cfg)
                llm_instances.append(svc)
                return svc

            monkeypatch.setattr(
                mcp_engine_manager, "build_llm_service", _deterministic_builder
            )
            monkeypatch.setattr(
                mcp_embeddings, "build_llm_service", _deterministic_builder
            )

        opened_for_harness: list[GraphRAG] = []
        manager = EngineManager(cfg)

        orig_get_rag = manager.get_rag

        async def _wrapped_get_rag(r: str | None = None) -> GraphRAG:
            rag = await orig_get_rag(realm=r)
            if all(x is not rag for x in opened_for_harness):
                opened_for_harness.append(rag)
            if all(x is not rag for x in all_opened_rags):
                all_opened_rags.append(rag)
            return rag

        manager.get_rag = _wrapped_get_rag
        managers.append(manager)

        server = create_mcp_server(mcp_cfg=cfg, manager=manager)
        return MCPTestHarness(
            config=cfg,
            manager=manager,
            server=server,
            llm_instances=llm_instances,
            opened_rags=opened_for_harness,
        )

    yield _make

    await cleanup_managers(managers, all_opened_rags)


@pytest_asyncio.fixture
async def mcp_harness(
    mcp_harness_factory: Callable[..., MCPTestHarness],
) -> MCPTestHarness:
    """Default engineering-preset MCP harness bound to a fresh PostgreSQL realm."""
    return mcp_harness_factory()
