"""Runtime configuration and workspace path sandboxing for post-graph-rag-mcp."""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EmbeddingBackend = Literal["fastembed", "openai"]

KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072,
}

_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_PGDATA_PATH = "~/.local/share/post-graph-rag-mcp/pgdata"
_VALID_PRESETS = frozenset({"engineering", "general_memory"})


def _nonblank_env(*names: str) -> str | None:
    """Return the first set, non-whitespace value among *names*, or None."""
    for name in names:
        val = os.getenv(name)
        if val is not None and val.strip():
            return val.strip()
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def derive_workspace_realm(cwd: Path | None = None) -> str:
    """Derive a deterministic PostgreSQL realm identifier from the repository root.

    Walks upward from *cwd* (defaulting to ``Path.cwd()``) looking for a ``.git``
    directory or gitdir pointer file. Falls back to the leaf name of *cwd* when outside
    a Git checkout, normalizes to ``[a-z0-9_]+``, prefixes with ``repo_``, and
    caps total length at 40 characters (returning ``"repo_default"`` if empty).
    """
    start = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    repo_name: str | None = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            repo_name = candidate.name
            break

    raw_name = repo_name if repo_name else start.name
    slug = re.sub(r"[^a-z0-9]+", "_", (raw_name or "").lower()).strip("_")
    if not slug:
        return "repo_default"

    realm = f"repo_{slug}"[:40].rstrip("_")
    return realm if len(realm) > len("repo_") else "repo_default"


def _resolve_embedding_backend(raw: str | None = None) -> EmbeddingBackend:
    choice = (raw if raw is not None else (os.getenv("MCP_EMBEDDING_BACKEND") or "")).strip().lower()
    if choice in ("", "auto"):
        has_fastembed = importlib.util.find_spec("fastembed") is not None
        if has_fastembed and not _nonblank_env("RAG_EMBEDDING_MODEL"):
            return "fastembed"
        if _nonblank_env("OPENAI_API_KEY", "RAG_EMBEDDING_API_KEY"):
            return "openai"
        return "fastembed"
    if choice == "fastembed":
        return "fastembed"
    if choice == "openai":
        return "openai"
    raise ValueError(
        f"Invalid MCP_EMBEDDING_BACKEND={choice!r}; expected 'auto', 'fastembed', or 'openai'."
    )


def _resolve_embedding_model(backend: EmbeddingBackend) -> str:
    if backend == "fastembed":
        return _nonblank_env("MCP_FASTEMBED_MODEL") or _DEFAULT_FASTEMBED_MODEL
    return _nonblank_env("RAG_EMBEDDING_MODEL") or _DEFAULT_OPENAI_EMBED_MODEL


def _resolve_embedding_dim(model: str, backend: EmbeddingBackend) -> int:
    override = _nonblank_env("RAG_EMBEDDING_DIM")
    if override is not None:
        dim = int(override)
        if dim <= 0:
            raise ValueError(f"RAG_EMBEDDING_DIM must be a positive integer, got {override!r}.")
        return dim
    fallback = 384 if backend == "fastembed" else 1536
    return KNOWN_EMBEDDING_DIMS.get(model, fallback)


def _resolve_allowed_roots(cwd: Path | None = None) -> list[Path]:
    raw = _nonblank_env("MCP_ALLOWED_ROOTS")
    if raw is not None:
        roots = [
            Path(part.strip()).expanduser().resolve()
            for part in raw.split(os.pathsep)
            if part.strip()
        ]
        if roots:
            return roots
    base = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    return [base]


def _resolve_statement_cache_size() -> int | None:
    raw = _nonblank_env("RAG_STATEMENT_CACHE_SIZE")
    return int(raw) if raw is not None else None


@dataclass
class MCPServerConfig:
    """Configuration for the post-graph-rag MCP server."""

    postgres_uri: str | None = field(
        default_factory=lambda: _nonblank_env("POSTGRES_URI", "DATABASE_URL", "RAG_DB_URI")
    )
    realm: str = field(
        default_factory=lambda: _nonblank_env("RAG_REALM") or derive_workspace_realm()
    )
    default_space: str = field(
        default_factory=lambda: _nonblank_env("RAG_SPACE") or "default"
    )
    preset: str = field(
        default_factory=lambda: _nonblank_env("MCP_MEMORY_PRESET") or "engineering"
    )
    embedding_backend: EmbeddingBackend = field(
        default_factory=_resolve_embedding_backend
    )
    embedding_model: str = ""
    embedding_dim: int = 0
    autonomous_llm_enabled: bool = field(
        default_factory=lambda: bool(_nonblank_env("OPENAI_API_KEY", "RAG_API_KEY"))
    )
    contradiction_detection: bool = field(
        default_factory=lambda: (
            _env_bool("RAG_CONTRADICTION_DETECTION", False)
            and bool(_nonblank_env("OPENAI_API_KEY", "RAG_API_KEY"))
        )
    )
    allowed_roots: list[Path] = field(default_factory=_resolve_allowed_roots)
    pgdata_dir: Path = field(
        default_factory=lambda: Path(
            _nonblank_env("MCP_PGDATA_DIR") or _DEFAULT_PGDATA_PATH
        ).expanduser().resolve()
    )
    pool_min_size: int = 1
    pool_max_size: int = 4
    schema_per_realm: bool = True
    statement_cache_size: int | None = field(default_factory=_resolve_statement_cache_size)

    def __post_init__(self) -> None:
        if self.postgres_uri is not None:
            self.postgres_uri = self.postgres_uri.strip() or None

        self.realm = (self.realm or "").strip() or derive_workspace_realm()
        self.default_space = (self.default_space or "").strip() or "default"
        self.preset = (self.preset or "").strip() or "engineering"
        if self.preset not in _VALID_PRESETS:
            raise ValueError(
                f"Unknown preset {self.preset!r}. Expected one of: "
                f"{', '.join(sorted(_VALID_PRESETS))}."
            )

        if self.embedding_backend not in ("fastembed", "openai"):
            self.embedding_backend = _resolve_embedding_backend(str(self.embedding_backend))

        if not (self.embedding_model or "").strip():
            self.embedding_model = _resolve_embedding_model(self.embedding_backend)
        else:
            self.embedding_model = self.embedding_model.strip()

        if self.embedding_dim <= 0:
            self.embedding_dim = _resolve_embedding_dim(
                self.embedding_model, self.embedding_backend
            )

        if not self.autonomous_llm_enabled:
            self.contradiction_detection = False

        if self.allowed_roots:
            self.allowed_roots = [Path(p).expanduser().resolve() for p in self.allowed_roots]
        else:
            self.allowed_roots = [Path.cwd().resolve()]

        self.pgdata_dir = Path(self.pgdata_dir).expanduser().resolve()

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> MCPServerConfig:
        """Construct an :class:`MCPServerConfig` from current environment variables."""
        backend = _resolve_embedding_backend()
        model = _resolve_embedding_model(backend)
        dim = _resolve_embedding_dim(model, backend)
        autonomous = bool(_nonblank_env("OPENAI_API_KEY", "RAG_API_KEY"))
        contradiction = _env_bool("RAG_CONTRADICTION_DETECTION", False) and autonomous
        pool_min = int(_nonblank_env("RAG_POOL_MIN_SIZE") or "1")
        pool_max = int(_nonblank_env("RAG_POOL_MAX_SIZE") or "4")
        schema_per_realm = _env_bool("RAG_SCHEMA_PER_REALM", True)

        return cls(
            postgres_uri=_nonblank_env("POSTGRES_URI", "DATABASE_URL", "RAG_DB_URI"),
            realm=_nonblank_env("RAG_REALM") or derive_workspace_realm(cwd),
            default_space=_nonblank_env("RAG_SPACE") or "default",
            preset=_nonblank_env("MCP_MEMORY_PRESET") or "engineering",
            embedding_backend=backend,
            embedding_model=model,
            embedding_dim=dim,
            autonomous_llm_enabled=autonomous,
            contradiction_detection=contradiction,
            allowed_roots=_resolve_allowed_roots(cwd),
            pgdata_dir=Path(
                _nonblank_env("MCP_PGDATA_DIR") or _DEFAULT_PGDATA_PATH
            ).expanduser().resolve(),
            pool_min_size=pool_min,
            pool_max_size=pool_max,
            schema_per_realm=schema_per_realm,
            statement_cache_size=_resolve_statement_cache_size(),
        )

    def validate_ingest_path(self, candidate: str | Path) -> Path:
        """Resolve *candidate* and verify that it resides within ``allowed_roots``.

        Resolves relative paths and symlinks before checking containment so
        ``../`` traversal and symlink escapes outside ``allowed_roots`` raise
        :class:`PermissionError`.
        """
        raw = str(candidate).strip() if candidate is not None else ""
        if not raw:
            raise PermissionError("Ingest path must not be empty.")

        resolved = Path(raw).expanduser().resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots):
            allowed_str = ", ".join(str(r) for r in self.allowed_roots)
            raise PermissionError(
                f"Refusing to access '{candidate}' (resolved to '{resolved}'): "
                f"path is outside allowed workspace roots [{allowed_str}]."
            )
        return resolved
