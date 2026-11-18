"""Embedding and LLM service adapters for zero-key local ONNX and hybrid modes."""

from __future__ import annotations

import asyncio
import collections
import os
from collections.abc import AsyncIterator
from typing import Any

from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import EmbeddingError, LLMError
from post_graph_rag.llm import LLMService
from pydantic import BaseModel

from post_graph_rag_mcp.config import MCPServerConfig

ZERO_KEY_CHAT_DISABLED_MESSAGE = (
    "Chat LLM is disabled in Zero-Key Host-Agent Mode. "
    "Pass structured triples/keywords from your IDE agent, or set OPENAI_API_KEY "
    "to enable autonomous LLM mode."
)


class _FastEmbedBackendMixin:
    """Lazy thread-offloaded FastEmbed ONNX vector computation shared by service classes."""

    config: RAGConfig
    model_name: str
    _embedder: Any
    _embed_lock: asyncio.Lock

    def _setup_fastembed_state(self, model_name: str | None = None) -> None:
        self.model_name = (model_name or self.config.embedding_model).strip()
        self._embedder = None
        self._embed_lock = asyncio.Lock()

    def _init_model(self) -> Any:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbeddingError(
                "Local ONNX embeddings require the 'fastembed' package. Install with:\n"
                '  pip install "post-graph-rag-mcp[local-embed]"\n'
                "or set OPENAI_API_KEY and MCP_EMBEDDING_BACKEND=openai."
            ) from exc

        try:
            return TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to initialize FastEmbed model '{self.model_name}': {exc}"
            ) from exc

    async def _ensure_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        async with self._embed_lock:
            if self._embedder is None:
                self._embedder = await asyncio.to_thread(self._init_model)
        return self._embedder

    def _compute_embeddings_sync(self, embedder: Any, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, int(self.config.embedding_batch_size))
        try:
            raw_vectors = list(embedder.embed(texts, batch_size=batch_size))
        except Exception as exc:
            raise EmbeddingError(
                f"FastEmbed inference failed for model '{self.model_name}' "
                f"(n={len(texts)}): {exc}"
            ) from exc

        if len(raw_vectors) != len(texts):
            raise EmbeddingError(
                f"FastEmbed model '{self.model_name}' returned {len(raw_vectors)} "
                f"vectors for {len(texts)} inputs."
            )

        out: list[list[float]] = []
        expected_dim = self.config.embedding_dim
        for vec in raw_vectors:
            values = [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
            if len(values) != expected_dim:
                raise EmbeddingError(
                    f"Embedding model '{self.model_name}' returned {len(values)} "
                    f"dimensions but RAGConfig.embedding_dim is {expected_dim}."
                )
            out.append(values)
        return out

    async def get_embedding(self, text: str) -> list[float]:
        embedder = await self._ensure_embedder()
        vecs = await asyncio.to_thread(self._compute_embeddings_sync, embedder, [text])
        return vecs[0]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = await self._ensure_embedder()
        return await asyncio.to_thread(self._compute_embeddings_sync, embedder, list(texts))


class FastEmbedLLMService(_FastEmbedBackendMixin, LLMService):
    """Zero-key local ONNX embedding service with chat completions explicitly disabled.

    Defers importing ``fastembed`` and loading ONNX weights until the first
    embedding request so MCP ``initialize`` handshakes finish in sub-millisecond
    wall time.
    """

    def __init__(self, config: RAGConfig, model_name: str | None = None) -> None:
        self.config = config
        self.client = None  # type: ignore[assignment]
        self.served = collections.Counter()
        self._setup_fastembed_state(model_name=model_name)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: type[BaseModel] | None = None,
        role: str | None = None,
    ) -> Any:
        raise LLMError(ZERO_KEY_CHAT_DISABLED_MESSAGE)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        role: str | None = None,
    ) -> AsyncIterator[str]:
        raise LLMError(ZERO_KEY_CHAT_DISABLED_MESSAGE)
        if False:  # pragma: no cover
            yield ""


class HybridFastEmbedOpenAILLMService(_FastEmbedBackendMixin, LLMService):
    """Local FastEmbed embeddings paired with OpenAI-compatible chat completions."""

    def __init__(self, config: RAGConfig, model_name: str | None = None) -> None:
        super().__init__(config)
        self._setup_fastembed_state(model_name=model_name)


def _apply_env_api_key_fallbacks(rag_cfg: RAGConfig, *, for_embeddings: bool) -> None:
    if rag_cfg.api_key and rag_cfg.api_key != "EMPTY":
        return
    candidates = (
        ("OPENAI_API_KEY", "RAG_EMBEDDING_API_KEY", "RAG_API_KEY")
        if for_embeddings
        else ("OPENAI_API_KEY", "RAG_API_KEY")
    )
    for var in candidates:
        val = (os.getenv(var) or "").strip()
        if val:
            rag_cfg.api_key = val
            return


def build_llm_service(mcp_cfg: MCPServerConfig, rag_cfg: RAGConfig) -> LLMService:
    """Build the appropriate :class:`LLMService` for *mcp_cfg* and *rag_cfg*."""
    if mcp_cfg.embedding_backend == "fastembed":
        if not mcp_cfg.autonomous_llm_enabled:
            return FastEmbedLLMService(rag_cfg, model_name=mcp_cfg.embedding_model)
        _apply_env_api_key_fallbacks(rag_cfg, for_embeddings=False)
        return HybridFastEmbedOpenAILLMService(rag_cfg, model_name=mcp_cfg.embedding_model)

    _apply_env_api_key_fallbacks(rag_cfg, for_embeddings=True)
    return LLMService(rag_cfg)
