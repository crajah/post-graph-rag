"""Exception hierarchy for post-graph-rag.

These exist so that misconfiguration and upstream failures surface to the caller
instead of silently degrading retrieval into an unranked ``LIMIT n`` scan.
"""


class RAGError(Exception):
    """Base class for all post-graph-rag errors."""


class SchemaError(RAGError):
    """Raised when the PostgreSQL graph schema is missing, unusable, or mismatched.

    The most common cause is the ``vector`` extension not being installed, or an
    existing table whose embedding column dimensionality differs from
    ``RAGConfig.embedding_dim``.
    """


class EmbeddingError(RAGError):
    """Raised when an embedding cannot be produced for a text."""


class LLMError(RAGError):
    """Raised when the LLM endpoint fails or returns an unusable response."""


class ExtractionError(RAGError):
    """Raised when entity/triple extraction fails.

    Deliberately *not* recoverable by fabricating generic edges: writing
    placeholder relations into a knowledge graph is worse than writing nothing,
    because it is indistinguishable from real extracted structure once stored.
    """
