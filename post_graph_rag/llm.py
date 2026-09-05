"""LLM and Embedding service wrapper for OpenAI-compatible endpoints."""
import asyncio
import collections
import hashlib
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import EmbeddingError, LLMError

logger = logging.getLogger(__name__)

# Conditions worth retrying or failing over to another model, rather than
# aborting: rate limits, exhausted credits, and upstream server errors.
RETRYABLE_STATUS = {402, 408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_MARKERS = (
    "run out of credits", "rate limit", "overloaded", "timeout", "timed out",
    "temporarily unavailable", "service unavailable", "capacity",
    # Transport-level failures. Concurrent indexing can momentarily exhaust a
    # local router's connection pool, which surfaces as a bare connection error
    # rather than an HTTP status.
    "connection error", "connection reset", "connection aborted",
    "connection refused", "server disconnected", "remote end closed",
    "temporary failure in name resolution", "broken pipe",
)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status in RETRYABLE_STATUS:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in RETRYABLE_MARKERS)


class LLMService:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key
        )
        # Which model actually served each successful call, not which one was
        # configured. With fallbacks enabled those differ, and silently so: a
        # router cooldown mid-run can leave half a graph extracted by the
        # primary model and half by a fallback, which is indistinguishable
        # afterwards from a graph the primary built alone. Extraction variance
        # is large enough that this changes what a measurement means.
        self.served: collections.Counter = collections.Counter()

    # -------------------------------------------------------------- embeddings

    def _encoding_format_kwargs(self) -> Dict[str, Any]:
        """Pin the wire format for embeddings, or defer to the SDK when unset.

        Left alone, the OpenAI SDK negotiates 'base64' by itself. Gateways that
        front non-OpenAI providers reject the parameter outright rather than
        ignoring it — litellm in front of Vertex AI fails every embedding call
        with UnsupportedParamsError — so the portable move is to state 'float'.
        """
        fmt = self.config.embedding_encoding_format
        return {"encoding_format": fmt} if fmt else {}

    async def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for a text.

        Raises on failure unless ``config.allow_embedding_fallback`` is set. A
        fallback vector shares a table with real embeddings but lives in an
        unrelated geometry, so distances between the two are meaningless — silently
        substituting one corrupts the index rather than degrading it.
        """
        async def attempt(_model: str):
            # Embedding models are not interchangeable across vector spaces, so
            # only the configured one is retried — never failed over.
            response = await self.client.embeddings.create(
                input=text,
                model=self.config.embedding_model,
                **self._encoding_format_kwargs(),  # type: ignore[arg-type]
            )
            return response.data[0].embedding

        try:
            vec = await self._retry_same(attempt, "Embedding")
        except Exception as e:
            if not self.config.allow_embedding_fallback:
                raise EmbeddingError(
                    f"Embedding request to {self.config.api_base} "
                    f"(model={self.config.embedding_model}) failed: {e}. "
                    f"Set RAGConfig.allow_embedding_fallback=True to use local/deterministic "
                    f"vectors instead, but note they are not comparable with API embeddings."
                ) from e
            logger.warning("Embedding API failed (%s); using local fallback vector.", e)
            return self._generate_local_fallback_embedding(text, self.config.embedding_dim)

        if len(vec) != self.config.embedding_dim:
            raise EmbeddingError(
                f"Embedding model '{self.config.embedding_model}' returned {len(vec)} "
                f"dimensions but RAGConfig.embedding_dim is {self.config.embedding_dim}. "
                f"The vector column will reject this write."
            )
        return vec

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed several texts in one request.

        Indexing a chunk needs one embedding per extracted entity. Issued one at
        a time those round trips dominate indexing latency, and the OpenAI
        embeddings endpoint accepts a batch natively.
        """
        if not texts:
            return []

        async def attempt(_model: str):
            response = await self.client.embeddings.create(
                input=texts,
                model=self.config.embedding_model,
                **self._encoding_format_kwargs(),  # type: ignore[arg-type]
            )
            # The API may return results out of order; index is authoritative.
            ordered = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]

        try:
            vecs = await self._retry_same(attempt, "Batch embedding")
        except Exception as e:
            if not self.config.allow_embedding_fallback:
                raise EmbeddingError(
                    f"Batch embedding request to {self.config.api_base} "
                    f"(model={self.config.embedding_model}, n={len(texts)}) failed: {e}"
                ) from e
            logger.warning("Batch embedding failed (%s); using local fallback vectors.", e)
            return [self._generate_local_fallback_embedding(t, self.config.embedding_dim) for t in texts]

        if len(vecs) != len(texts):
            raise EmbeddingError(
                f"Embedding model returned {len(vecs)} vectors for {len(texts)} inputs."
            )
        for vec in vecs:
            if len(vec) != self.config.embedding_dim:
                raise EmbeddingError(
                    f"Embedding model '{self.config.embedding_model}' returned {len(vec)} "
                    f"dimensions but RAGConfig.embedding_dim is {self.config.embedding_dim}."
                )
        return vecs

    def _generate_local_fallback_embedding(self, text: str, dim: int) -> List[float]:
        """Produce a fallback vector when the embedding API is unavailable.

        Tries local transformer models first, then falls back to a deterministic
        term-hash unit vector. Determinism is load-bearing: vectors written at
        index time must still match at query time in a later process.
        """
        for loader in (self._try_fastembed, self._try_sentence_transformers):
            vec = loader(text)
            if vec is None:
                continue
            if len(vec) == dim:
                return vec
            logger.warning(
                "Local embedding model returned %d dims, but embedding_dim is %d; skipping.",
                len(vec), dim
            )

        return self._term_hash_embedding(text, dim)

    @staticmethod
    def _try_fastembed(text: str) -> Optional[List[float]]:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
            model = TextEmbedding()
            return [float(x) for x in next(model.embed([text]))]
        except Exception:
            return None

    @staticmethod
    def _try_sentence_transformers(text: str) -> Optional[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return [float(x) for x in model.encode(text).tolist()]
        except Exception:
            return None

    @staticmethod
    def _term_hash_embedding(text: str, dim: int) -> List[float]:
        """Deterministic term-frequency hash vector.

        Uses SHA-256 for the bucket index as well as the magnitude. Python's
        builtin ``hash()`` is salted per process (PYTHONHASHSEED), so using it
        here produced a different vector on every run — meaning nothing indexed
        could ever be retrieved after a restart.
        """
        words = [w.strip() for w in text.lower().split() if w.strip()]
        vec = [0.0] * dim
        if not words or dim <= 0:
            return vec

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket_seed = int.from_bytes(digest[:8], "big")
            for idx, byte in enumerate(digest):
                dim_idx = (bucket_seed + idx) % dim
                vec[dim_idx] += (byte - 128) / 128.0

        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    # ------------------------------------------------------------- completions

    def _deadline(self) -> float:
        """Monotonic instant after which a call stops retrying."""
        return time.monotonic() + max(0.0, self.config.retry_deadline_secs)

    async def _retry_same(self, attempt, what: str) -> Any:
        """Retry a single target with backoff, without switching models."""
        last: Optional[Exception] = None
        deadline = self._deadline()
        for tries in range(1, max(1, self.config.max_retries) + 1):
            try:
                return await attempt(self.config.embedding_model)
            except Exception as e:
                last = e
                if not _is_retryable(e):
                    raise
                logger.warning(
                    "%s attempt %d/%d failed (%s)", what, tries, self.config.max_retries, str(e)[:160]
                )
                if time.monotonic() >= deadline:
                    logger.warning("%s: retry deadline reached; giving up.", what)
                    break
                if tries < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_backoff_secs * tries)
        raise last if last else RuntimeError(f"{what} failed")

    def _model_candidates(self) -> List[str]:
        """Primary model first, then declared fallbacks, preserving order."""
        seen, models = set(), []
        for name in [self.config.model, *self.config.fallback_models]:
            if name and name not in seen:
                seen.add(name)
                models.append(name)
        return models

    async def _with_failover(self, attempt, what: str) -> Any:
        """Run ``attempt(model)`` across candidate models with backoff.

        Retries the same model for transient failures, then moves to the next
        candidate. Non-retryable errors abort immediately so genuine mistakes
        (bad request, unknown model) are not masked by a long retry loop.
        """
        last: Optional[Exception] = None
        deadline = self._deadline()
        for model in self._model_candidates():
            for tries in range(1, max(1, self.config.max_retries) + 1):
                try:
                    result = await attempt(model)
                    self.served[model] += 1
                    return result
                except Exception as e:
                    last = e
                    if not _is_retryable(e):
                        raise LLMError(f"{what} failed for model '{model}': {e}") from e
                    logger.warning(
                        "%s: model '%s' attempt %d/%d failed (%s)",
                        what, model, tries, self.config.max_retries, str(e)[:160],
                    )
                    # A sustained outage should surface fast rather than burning
                    # retries x models x backoff on every single call.
                    if time.monotonic() >= deadline:
                        raise LLMError(
                            f"{what} exceeded the {self.config.retry_deadline_secs}s retry "
                            f"deadline at {self.config.api_base}: {e}"
                        ) from e
                    if tries < self.config.max_retries:
                        await asyncio.sleep(self.config.retry_backoff_secs * tries)
            logger.warning("%s: giving up on model '%s'; trying next candidate.", what, model)

        raise LLMError(
            f"{what} failed for all models {self._model_candidates()} at "
            f"{self.config.api_base}: {last}"
        ) from last

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[BaseModel]] = None
    ) -> Any:
        """Call the LLM completion endpoint, optionally enforcing structured output.

        Retries and fails over across ``fallback_models``. Raises
        :class:`LLMError` when every candidate is exhausted. Previously this
        returned ``""`` or a string stitched together from the prompt, which
        reached the caller looking like a real model answer.
        """
        if response_format is not None:
            async def structured(model: str):
                response = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format=response_format
                )
                return response.choices[0].message.parsed

            try:
                parsed = await self._with_failover(structured, "Structured completion")
                if parsed is not None:
                    return parsed
                logger.warning("Structured output returned no parsed value; retrying unstructured.")
            except Exception as e:
                logger.warning("Structured output unsupported or exhausted (%s); retrying unstructured.", str(e)[:160])

        async def plain(model: str):
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""

        return await self._with_failover(plain, "Chat completion")

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream completion content chunks.

        Referenced by the engine's ``QueryParam(stream=True)`` path, which
        previously raised AttributeError because this method did not exist.
        """
        async def open_stream(model: str):
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                stream=True
            )

        # Failover applies to opening the stream. Once tokens have been yielded a
        # retry would duplicate output, so mid-stream failures propagate.
        stream = await self._with_failover(open_stream, "Streaming chat completion")

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            raise LLMError(
                f"Streaming chat completion failed mid-stream at {self.config.api_base}: {e}"
            ) from e
