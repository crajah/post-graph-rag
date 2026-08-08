"""Configuration dataclass for post-graph-rag."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Placeholder accepted by local OpenAI-compatible servers (vLLM, LiteLLM, Ollama)
# that do not authenticate. Real deployments set OPENAI_API_KEY.
_LOCAL_API_KEY_PLACEHOLDER = "EMPTY"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class RAGConfig:
    """Runtime configuration.

    Every default is resolved via ``default_factory`` so the environment is read
    when a config is constructed rather than when this module is first imported.
    """

    api_base: str = field(default_factory=lambda: _env("OPENAI_API_BASE", "http://localhost:4000/v1"))
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", _LOCAL_API_KEY_PLACEHOLDER))
    model: str = field(default_factory=lambda: _env("RAG_MODEL", "DeepSeek-V3.2"))
    embedding_model: str = field(default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: int(_env("RAG_EMBEDDING_DIM", "1536")))
    db_uri: str = field(default_factory=lambda: _env("POSTGRES_URI", "postgresql://localhost:5432/postgres"))
    realm: str = field(default_factory=lambda: _env("RAG_REALM", "default"))
    space: str = field(default_factory=lambda: _env("RAG_SPACE", "default"))

    # Models to try, in order, when `model` fails with a retryable error
    # (rate limits, exhausted credits, upstream 5xx). Routers fronting several
    # providers commonly exhaust one account's credits mid-run while other
    # models stay available, which otherwise aborts a long indexing job.
    fallback_models: List[str] = field(
        default_factory=lambda: [
            m.strip() for m in _env("RAG_FALLBACK_MODELS", "").split(",") if m.strip()
        ]
    )
    # Attempts per model before moving to the next one.
    max_retries: int = field(default_factory=lambda: int(_env("RAG_MAX_RETRIES", "5")))
    retry_backoff_secs: float = field(default_factory=lambda: float(_env("RAG_RETRY_BACKOFF", "2.0")))
    # Wall-clock ceiling for a single call including all retries and failovers.
    # Without it, retries x models x backoff means a sustained outage burns
    # enormous time per call before giving up — a 12-community build spent 38
    # minutes retrying to produce nothing. Retrying is for transient blips; a
    # sustained outage should surface quickly.
    retry_deadline_secs: float = field(
        default_factory=lambda: float(_env("RAG_RETRY_DEADLINE", "120"))
    )

    # ----------------------------------------------------------- extraction
    # Extra passes asking the model what it missed. Single-pass extraction
    # under-recalls on dense text; each pass costs one more LLM call per chunk.
    gleaning_passes: int = field(default_factory=lambda: int(_env("RAG_GLEANING_PASSES", "1")))

    # Override the extraction system prompt wholesale. None uses the built-in
    # prompt, rendered from `entity_types` and `predicate_vocabulary`.
    extraction_prompt: Optional[str] = None

    # Preferred entity types. Empty uses the library defaults.
    entity_types: List[str] = field(
        default_factory=lambda: [t.strip() for t in _env("RAG_ENTITY_TYPES", "").split(",") if t.strip()]
    )

    # Preferred predicates. When set, the model is steered onto this list and
    # extracted predicates are snapped to it where they clearly match. Empty
    # leaves predicates free-form, which maximises fidelity but produces a
    # vocabulary too sparse to query by relation type.
    predicate_vocabulary: List[str] = field(
        default_factory=lambda: [p.strip() for p in _env("RAG_PREDICATE_VOCABULARY", "").split(",") if p.strip()]
    )
    # Explicit synonym map applied after normalisation, e.g. {"collaborated_with": "worked_with"}.
    predicate_aliases: Dict[str, str] = field(default_factory=dict)

    # Drop relations the text asserts do NOT hold, instead of storing them with
    # negated=true. Off by default: the negation is itself information.
    drop_negated_relations: bool = field(
        default_factory=lambda: _env("RAG_DROP_NEGATED", "0").lower() in ("1", "true", "yes")
    )
    # Reject possessive role phrases ("Babbage's father") as entities. Off by
    # default: on real prose the same rule also rejects legitimate named things
    # such as "Ampere's force law", so about half its hits are false positives.
    reject_possessive_entities: bool = field(
        default_factory=lambda: _env("RAG_REJECT_POSSESSIVE_ENTITIES", "0").lower() in ("1", "true", "yes")
    )

    # Discard relations the model is less sure about than this.
    min_relation_confidence: float = field(
        default_factory=lambda: float(_env("RAG_MIN_RELATION_CONFIDENCE", "0.0"))
    )

    # Chunk sizing used by GraphRAG.index_text and the default chunker.
    chunk_chars: int = field(default_factory=lambda: int(_env("RAG_CHUNK_CHARS", "2000")))
    chunk_overlap_chars: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "200")))

    # Chunks whose LLM and embedding calls run concurrently. Indexing is almost
    # entirely network-bound, so this is the dominant lever on wall-clock time.
    # Chunks are processed in batches of this size: within a batch they run in
    # parallel, and each batch sees the entities discovered by earlier batches,
    # so coreference context is threaded at batch granularity rather than lost.
    # Set to 1 for strict sequential indexing and per-chunk context.
    max_concurrent_chunks: int = field(
        default_factory=lambda: int(_env("RAG_MAX_CONCURRENT_CHUNKS", "4"))
    )

    # Cap on canonical entity names passed back into extraction as context.
    context_entity_limit: int = field(default_factory=lambda: int(_env("RAG_CONTEXT_ENTITY_LIMIT", "40")))

    # Pull in chunks that mention a retrieved entity, not just chunks that match
    # the query vector directly. This is what makes entity hits surface their
    # supporting passages across documents.
    expand_chunks_via_mentions: bool = field(
        default_factory=lambda: _env("RAG_EXPAND_VIA_MENTIONS", "1").lower() in ("1", "true", "yes")
    )

    # --------------------------------------------------------- communities
    # Corpus-level questions ("what are the main themes?") are answered from
    # summaries of clustered subgraphs, not from individual passages.
    # Smallest cluster worth summarising. Pairs and singletons produce reports
    # that say nothing their members' own descriptions do not.
    community_min_size: int = field(default_factory=lambda: int(_env("RAG_COMMUNITY_MIN_SIZE", "3")))
    # Higher values yield more, smaller communities (Leiden only).
    community_resolution: float = field(default_factory=lambda: float(_env("RAG_COMMUNITY_RESOLUTION", "1.0")))
    # Cap on communities summarised per build; each costs one LLM call.
    max_communities: int = field(default_factory=lambda: int(_env("RAG_MAX_COMMUNITIES", "64")))
    # Override the community report prompt.
    community_report_prompt: Optional[str] = None

    # How community reports are ranked for global retrieval. Pure similarity
    # lets a small, tightly-worded niche cluster outrank the central theme on a
    # broad question ("what are the main themes?"), because a narrow cluster's
    # summary can sit closer to a short query than a broad one's. Blending in
    # the report's own importance rating and its size corrects for that.
    # Weights are relative; set importance and size to 0.0 for pure similarity.
    community_weight_similarity: float = field(
        default_factory=lambda: float(_env("RAG_COMMUNITY_W_SIMILARITY", "1.0"))
    )
    community_weight_importance: float = field(
        default_factory=lambda: float(_env("RAG_COMMUNITY_W_IMPORTANCE", "0.25"))
    )
    community_weight_size: float = field(
        default_factory=lambda: float(_env("RAG_COMMUNITY_W_SIZE", "0.15"))
    )
    # Candidates fetched before re-ranking, as a multiple of top_k.
    community_candidate_multiplier: int = field(
        default_factory=lambda: int(_env("RAG_COMMUNITY_CANDIDATE_MULTIPLIER", "3"))
    )
    # Weight assigned to a denied relation when clustering. Negated relations
    # still connect their endpoints topically, but less strongly.
    negated_relation_weight: float = field(
        default_factory=lambda: float(_env("RAG_NEGATED_RELATION_WEIGHT", "0.3"))
    )

    # Give each realm its own PostgreSQL schema instead of sharing one set of
    # tables filtered by a realm column. Off by default for backwards
    # compatibility, but recommended: with shared tables the first realm to
    # create `entities` fixes the embedding column width for every other realm.
    schema_per_realm: bool = field(
        default_factory=lambda: _env("RAG_SCHEMA_PER_REALM", "0").lower() in ("1", "true", "yes")
    )

    # Give relation edges their own embeddings so they can be retrieved by
    # similarity. Optional and off by default: relations are normally reached by
    # traversing out from a matched entity, and enabling this costs one extra
    # embedding call per triple at index time.
    embed_relations: bool = field(
        default_factory=lambda: _env("RAG_EMBED_RELATIONS", "0").lower() in ("1", "true", "yes")
    )

    # When the embedding API is unreachable, fall back to a local/deterministic
    # vector instead of raising. Off by default: fallback vectors are not
    # comparable with API embeddings, so mixing them silently breaks retrieval.
    allow_embedding_fallback: bool = field(
        default_factory=lambda: _env("RAG_ALLOW_EMBEDDING_FALLBACK", "0").lower() in ("1", "true", "yes")
    )
