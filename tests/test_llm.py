"""Tests for LLMService: embedding determinism, error propagation, streaming."""
import subprocess
import sys
from typing import Any, Dict

import pytest

from post_graph_rag import LLMService, RAGConfig
from post_graph_rag.errors import EmbeddingError, LLMError


def _config(**kw):
    params: Dict[str, Any] = dict(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    params.update(kw)
    return RAGConfig(**params)


def test_term_hash_embedding_is_deterministic_within_process():
    svc = LLMService(_config())
    a = svc._term_hash_embedding("zeus is king of the gods", 16)
    b = svc._term_hash_embedding("zeus is king of the gods", 16)
    assert a == b


def test_term_hash_embedding_is_deterministic_across_processes():
    """The regression that mattered: builtin hash() is salted per process.

    Vectors written at index time must still match at query time after a
    restart, so this runs two fresh interpreters with different hash seeds.
    """
    code = (
        "from post_graph_rag import LLMService, RAGConfig;"
        "svc = LLMService(RAGConfig(api_base='http://localhost:9/v1', api_key='k', embedding_dim=16));"
        "print(svc._term_hash_embedding('zeus is king of the gods', 16))"
    )
    outs = []
    for seed in ("1", "999"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1]


def test_term_hash_embedding_is_normalised_and_correct_width():
    svc = LLMService(_config())
    vec = svc._term_hash_embedding("alpha beta gamma", 16)
    assert len(vec) == 16
    assert abs(sum(x * x for x in vec) ** 0.5 - 1.0) < 1e-9


def test_term_hash_embedding_empty_text():
    svc = LLMService(_config())
    assert svc._term_hash_embedding("", 16) == [0.0] * 16


@pytest.mark.asyncio
async def test_get_embedding_raises_when_fallback_disabled():
    """A fallback vector is not comparable with API embeddings, so the default
    is to fail rather than silently corrupt the index."""
    svc = LLMService(_config(allow_embedding_fallback=False))
    with pytest.raises(EmbeddingError):
        await svc.get_embedding("anything")


@pytest.mark.asyncio
async def test_get_embedding_falls_back_when_enabled():
    svc = LLMService(_config(allow_embedding_fallback=True))
    vec = await svc.get_embedding("zeus")
    assert len(vec) == 16


@pytest.mark.asyncio
async def test_chat_completion_raises_instead_of_fabricating():
    """Previously this returned '' or text stitched from the prompt, which
    reached the caller looking like a real model answer."""
    svc = LLMService(_config())
    with pytest.raises(LLMError):
        await svc.chat_completion([{"role": "user", "content": "User Question:\n- a\n- b"}])


@pytest.mark.asyncio
async def test_chat_completion_stream_exists_and_raises_on_failure():
    """Regression: this method was called by the engine but never defined."""
    svc = LLMService(_config())
    assert hasattr(svc, "chat_completion_stream")
    with pytest.raises(LLMError):
        async for _ in svc.chat_completion_stream([{"role": "user", "content": "hi"}]):
            pass
