"""Shared configuration for the examples.

Every example reads the same environment variables so you can point them all
at your own router and database once:

    export OPENAI_API_KEY=...
    export OPENAI_API_BASE=https://your-router/v1     # any OpenAI-compatible endpoint
    export POSTGRES_URI=postgresql://localhost:5432/postgres
"""
import os
import time

from post_graph_rag import RAGConfig


def fresh_realm(base: str) -> str:
    """A realm nobody has written to yet.

    The examples demonstrate ordering -- a watermark taken between two
    documents, a delta between two indexings -- and a realm carried over from
    a previous run has those documents already in it, with their original
    timestamps. The demo would then quietly show nothing. A per-run realm
    keeps every example repeatable and independent.
    """
    return f"{base}_{int(time.time())}"


def make_config(realm: str, **overrides) -> RAGConfig:
    """A config pointed at your router and database, with one realm per example."""
    params = dict(
        api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:4000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-set"),
        model=os.environ.get("PGR_MODEL", "gemini-3.6-flash"),
        embedding_model=os.environ.get("PGR_EMBED_MODEL", "gemini-embedding-001"),
        embedding_dim=int(os.environ.get("PGR_EMBED_DIM", "1536")),
        db_uri=os.environ.get("POSTGRES_URI",
                              "postgresql://localhost:5432/postgres"),
        realm=realm,
        schema_per_realm=True,
    )
    params.update(overrides)
    return RAGConfig(**params)


def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
