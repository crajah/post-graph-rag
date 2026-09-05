"""Tests for LLM retry logic and retryable error detection."""

import pytest

from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import LLMError
from post_graph_rag.llm import RETRYABLE_MARKERS, RETRYABLE_STATUS, LLMService, _is_retryable


def _rate_limit_error():
    """A retryable failure, shaped like the router's 429."""
    return Exception("Error code: 429 - rate limit exceeded")


def _config(**kw):
    params = dict(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    params.update(kw)
    return RAGConfig(**params)


class TestIsRetryable:
    def test_retryable_status_codes(self):
        for code in RETRYABLE_STATUS:
            exc = Exception("fail")
            exc.status_code = code
            assert _is_retryable(exc) is True

    def test_non_retryable_status(self):
        exc = Exception("fail")
        exc.status_code = 400
        assert _is_retryable(exc) is False

    def test_retryable_markers_in_message(self):
        for marker in RETRYABLE_MARKERS:
            exc = Exception(f"Error: {marker} encountered")
            assert _is_retryable(exc) is True, f"Marker '{marker}' should be retryable"

    def test_non_retryable_message(self):
        exc = Exception("bad request: invalid JSON")
        assert _is_retryable(exc) is False

    def test_status_on_response_attr(self):
        class FakeResponse:
            status_code = 429
        exc = Exception("rate limited")
        exc.response = FakeResponse()
        assert _is_retryable(exc) is True

    def test_no_status_no_markers(self):
        exc = Exception("something unknown")
        assert _is_retryable(exc) is False


class TestModelCandidates:
    def test_primary_only(self):
        svc = LLMService(_config(model="gpt-4"))
        assert svc._model_candidates() == ["gpt-4"]

    def test_with_fallbacks(self):
        svc = LLMService(_config(model="gpt-4", fallback_models=["gpt-3.5", "claude"]))
        assert svc._model_candidates() == ["gpt-4", "gpt-3.5", "claude"]

    def test_deduplicates(self):
        svc = LLMService(_config(model="gpt-4", fallback_models=["gpt-4", "gpt-3.5"]))
        candidates = svc._model_candidates()
        assert candidates == ["gpt-4", "gpt-3.5"]

    def test_empty_fallback_strings_ignored(self):
        svc = LLMService(_config(model="gpt-4", fallback_models=["", "gpt-3.5"]))
        candidates = svc._model_candidates()
        assert "" not in candidates


class TestTermHashEmbedding:
    def test_zero_dim(self):
        svc = LLMService(_config())
        vec = svc._term_hash_embedding("hello", 0)
        assert vec == []

    def test_single_word(self):
        svc = LLMService(_config())
        vec = svc._term_hash_embedding("zeus", 16)
        assert len(vec) == 16
        assert any(x != 0.0 for x in vec)

    def test_unit_length(self):
        svc = LLMService(_config())
        vec = svc._term_hash_embedding("some text here", 16)
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_different_texts_differ(self):
        svc = LLMService(_config())
        a = svc._term_hash_embedding("zeus olympian", 16)
        b = svc._term_hash_embedding("python programming", 16)
        assert a != b

    def test_whitespace_only_text(self):
        svc = LLMService(_config())
        vec = svc._term_hash_embedding("   ", 16)
        assert vec == [0.0] * 16


class TestBatchEmbeddingEmpty:
    async def test_get_embeddings_empty_list(self):
        svc = LLMService(_config())
        result = await svc.get_embeddings([])
        assert result == []


class TestEncodingFormatKwargs:
    def test_float_format(self):
        svc = LLMService(_config(embedding_encoding_format="float"))
        assert svc._encoding_format_kwargs() == {"encoding_format": "float"}

    def test_none_format(self):
        cfg = _config()
        cfg.embedding_encoding_format = None
        svc = LLMService(cfg)
        assert svc._encoding_format_kwargs() == {}

    def test_base64_format(self):
        svc = LLMService(_config(embedding_encoding_format="base64"))
        assert svc._encoding_format_kwargs() == {"encoding_format": "base64"}


# --------------------------------------------------- which model actually served

@pytest.mark.asyncio
async def test_served_records_the_model_that_succeeded():
    """Provenance, not configuration.

    A run that fails over builds its graph from more than one model, and
    afterwards that is indistinguishable from a run the primary served alone —
    unless it was counted at the time.
    """
    svc = LLMService(RAGConfig(model="primary", fallback_models=["backup"],
                               max_retries=2, retry_backoff_secs=0))
    calls = []

    async def attempt(model):
        calls.append(model)
        if model == "primary":
            raise _rate_limit_error()
        return "ok"

    assert await svc._with_failover(attempt, "test") == "ok"
    assert svc.served == {"backup": 1}
    assert "primary" not in svc.served


@pytest.mark.asyncio
async def test_served_counts_each_success_separately():
    svc = LLMService(RAGConfig(model="primary", max_retries=1, retry_backoff_secs=0))

    async def attempt(model):
        return "ok"

    for _ in range(3):
        await svc._with_failover(attempt, "test")
    assert svc.served["primary"] == 3


@pytest.mark.asyncio
async def test_served_stays_empty_when_everything_fails():
    """Nothing served means nothing counted — no phantom provenance."""
    svc = LLMService(RAGConfig(model="primary", fallback_models=["backup"],
                               max_retries=1, retry_backoff_secs=0))

    async def attempt(model):
        raise _rate_limit_error()

    with pytest.raises(LLMError):
        await svc._with_failover(attempt, "test")
    assert not svc.served
