"""Tests for post_graph_rag.errors — error hierarchy."""

import pytest

from post_graph_rag.errors import (
    EmbeddingError,
    ExtractionError,
    LLMError,
    RAGError,
    SchemaError,
)


class TestErrorHierarchy:
    def test_rag_error_is_exception(self):
        assert issubclass(RAGError, Exception)

    def test_schema_error_is_rag_error(self):
        assert issubclass(SchemaError, RAGError)

    def test_embedding_error_is_rag_error(self):
        assert issubclass(EmbeddingError, RAGError)

    def test_llm_error_is_rag_error(self):
        assert issubclass(LLMError, RAGError)

    def test_extraction_error_is_rag_error(self):
        assert issubclass(ExtractionError, RAGError)

    def test_catch_all(self):
        with pytest.raises(RAGError):
            raise SchemaError("test")
        with pytest.raises(RAGError):
            raise EmbeddingError("test")
        with pytest.raises(RAGError):
            raise LLMError("test")
        with pytest.raises(RAGError):
            raise ExtractionError("test")

    def test_message_preserved(self):
        err = SchemaError("missing pgvector")
        assert "missing pgvector" in str(err)
