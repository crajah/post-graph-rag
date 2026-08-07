"""Shared fixtures.

DB-backed tests run against a real PostgreSQL with pgvector and are skipped when
one is not reachable. They use ``schema_per_realm`` so every test realm gets its
own schema and cannot collide with real data.
"""
import os
import uuid
from typing import Any, Dict

import pytest
import pytest_asyncio

from post_graph_rag import GraphRAG, RAGConfig
from post_graph_rag.llm import LLMService

TEST_DSN = os.getenv("POSTGRES_TEST_URI", os.getenv("POSTGRES_URI", "postgresql://localhost:5432/postgres"))

# A tiny fixed vocabulary gives the fake embedder a real (if crude) semantic
# geometry, so relevance assertions mean something.
VOCAB = [
    "zeus", "olympian", "gods", "cronus", "rhea", "hera", "titans", "sea",
    "poseidon", "python", "programming", "language", "guido", "rossum",
    "created", "king",
]
VOCAB_DIM = len(VOCAB)


def fake_embed(text: str) -> list:
    """Bag-of-words unit vector over VOCAB."""
    lowered = (text or "").lower()
    vec = [float(lowered.count(term)) for term in VOCAB]
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        # Keep it non-zero so pgvector cosine distance stays defined.
        return [1.0 / (VOCAB_DIM ** 0.5)] * VOCAB_DIM
    return [x / norm for x in vec]


class FakeLLM(LLMService):
    """LLMService with the network calls replaced.

    Subclasses the real service so any signature drift in the methods under test
    shows up here rather than being silently mocked away.
    """

    def __init__(self, config, extraction=None, answer="fake answer", fail=False):
        self.config = config
        self.client = None
        self._extraction = extraction
        self._answer = answer
        self._fail = fail
        self.embed_calls = []
        self.batch_calls = []
        self.chat_calls = []
        self._structured_calls = 0

    async def get_embedding(self, text: str):
        from post_graph_rag.errors import EmbeddingError
        if self._fail:
            raise EmbeddingError("fake embedding failure")
        self.embed_calls.append(text)
        return fake_embed(text)

    async def get_embeddings(self, texts):
        from post_graph_rag.errors import EmbeddingError
        if self._fail:
            raise EmbeddingError("fake embedding failure")
        self.batch_calls.append(list(texts))
        self.embed_calls.extend(texts)
        return [fake_embed(t) for t in texts]

    async def chat_completion(self, messages, response_format=None):
        from post_graph_rag.errors import LLMError
        self.chat_calls.append((messages, response_format))
        if self._fail:
            raise LLMError("fake LLM failure")
        if response_format is not None and self._extraction is not None:
            # A list drives successive calls, which is how gleaning passes are
            # exercised: first call returns the base result, later calls the
            # "missed" records.
            if isinstance(self._extraction, list):
                idx = min(self._structured_calls, len(self._extraction) - 1)
                self._structured_calls += 1
                candidate = self._extraction[idx]
                return candidate if isinstance(candidate, response_format) else response_format()
            if isinstance(self._extraction, response_format):
                return self._extraction
            return response_format()
        if response_format is not None:
            return response_format()
        return self._answer

    async def chat_completion_stream(self, messages):
        from post_graph_rag.errors import LLMError
        if self._fail:
            raise LLMError("fake LLM failure")
        for token in self._answer.split():
            yield token + " "


def make_config(**overrides) -> RAGConfig:
    params: Dict[str, Any] = dict(
        api_base="http://localhost:9/v1",
        api_key="test",
        model="test-model",
        embedding_model="test-embed",
        embedding_dim=VOCAB_DIM,
        db_uri=TEST_DSN,
        realm="t" + uuid.uuid4().hex[:12],
        space="default",
        schema_per_realm=True,
    )
    params.update(overrides)
    return RAGConfig(**params)


async def _db_reachable(config: RAGConfig) -> bool:
    from post_graph import AsyncPostGraph
    client = AsyncPostGraph(dsn=config.db_uri)
    try:
        await client.connect()
        rows = await client._fetch("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        await client.close()
        return bool(rows)
    except Exception:
        return False


@pytest_asyncio.fixture
async def rag_factory():
    """Build GraphRAG instances bound to disposable realm schemas."""
    created = []

    async def _make(extraction=None, answer="fake answer", **cfg_overrides):
        config = make_config(**cfg_overrides)
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        rag = GraphRAG(config)
        rag.llm = FakeLLM(config, extraction=extraction, answer=answer)
        rag.extractor.llm_service = rag.llm
        await rag.initialize()
        created.append(rag)
        return rag

    yield _make

    for rag in created:
        try:
            await rag.store.client._execute(f'DROP SCHEMA IF EXISTS "{rag.config.realm}" CASCADE;')
            await rag.close()
        except Exception:
            pass
