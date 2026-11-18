"""Lazy GraphRAG lifecycle management, rule propagation, and Mode-A keyword fallback."""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any

from post_graph_rag import GraphRAG, LLMService, RAGConfig
from post_graph_rag.extractor import normalise_predicate

from post_graph_rag_mcp.config import MCPServerConfig
from post_graph_rag_mcp.embedded_pg import EmbeddedPostgresManager
from post_graph_rag_mcp.embeddings import build_llm_service
from post_graph_rag_mcp.presets import PresetPack, get_preset, list_presets

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "across",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "compare",
        "could",
        "currently",
        "describe",
        "did",
        "didn",
        "do",
        "does",
        "doesn",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "ever",
        "every",
        "explain",
        "few",
        "find",
        "for",
        "from",
        "further",
        "get",
        "give",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "it",
        "its",
        "itself",
        "just",
        "list",
        "may",
        "me",
        "might",
        "more",
        "most",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "please",
        "same",
        "shall",
        "she",
        "should",
        "show",
        "so",
        "some",
        "still",
        "such",
        "summarize",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "trace",
        "under",
        "until",
        "up",
        "us",
        "very",
        "was",
        "wasn",
        "we",
        "were",
        "weren",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
    }
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]*[A-Za-z0-9]|[A-Za-z0-9]+")


def _is_entity_like_token(token: str, position: int) -> bool:
    """Return True if *token* looks like a proper noun, identifier, acronym, or version/port."""
    if any(ch in token for ch in ("_", ".", "/", ":", "-")):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if len(token) > 1 and any(ch.isupper() for ch in token[1:]):
        return True
    if position > 0 and token[0].isupper():
        return True
    return token[0].isupper() and token.lower() not in STOPWORDS


def extract_fallback_keywords(question: str) -> tuple[list[str], list[str]]:
    """Deterministically extract ``(ll_keywords, hl_keywords)`` from *question*.

    Used in Mode A (Zero-Key Host-Agent Mode) whenever the caller leaves both
    ``entity_keywords`` and ``theme_keywords`` empty so ``GraphRAG.query_data()``
    never invokes ``GraphExtractor.extract_keywords()`` against an unconfigured
    chat LLM.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return ([], ["query"])

    raw_tokens = _TOKEN_PATTERN.findall(cleaned)
    entity_like: list[str] = []
    content_terms: list[str] = []
    seen_lower: set[str] = set()

    for idx, raw_tok in enumerate(raw_tokens):
        tok = raw_tok.strip(".,:;!?()[]{}\"'")
        if not tok:
            continue
        low = tok.lower()
        if low in STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        if low in seen_lower:
            continue
        seen_lower.add(low)
        if _is_entity_like_token(tok, idx):
            entity_like.append(tok)
        else:
            content_terms.append(tok)

    ll_keywords = (entity_like + content_terms)[:12]
    if not ll_keywords:
        ll_keywords = [cleaned]

    hl_keywords = [cleaned]
    return (ll_keywords, hl_keywords)


class EngineManager:
    """Manages lazy per-realm :class:`GraphRAG` instances and runtime supersession rules.

    Initialization is synchronous and sub-millisecond: neither PostgreSQL connection
    pools nor local ONNX embedding weights are opened until the first tool or resource
    invocation calls :meth:`get_rag`.
    """

    def __init__(
        self,
        mcp_cfg: MCPServerConfig | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.cfg: MCPServerConfig = mcp_cfg or MCPServerConfig.from_env()
        self._preset_name: str = self.cfg.preset
        get_preset(self._preset_name)

        self.extra_functional_predicates: set[str] = set()
        self.extra_exclusive_groups: list[set[str]] = []
        self.extra_vocabulary: set[str] = set()

        self._embedded = EmbeddedPostgresManager(self.cfg.pgdata_dir)
        self._engines: dict[str, GraphRAG] = {}
        self._lock = asyncio.Lock()
        self._llm_service_override: LLMService | None = llm_service

    @property
    def preset_name(self) -> str:
        """Name of the currently active base memory preset."""
        return self._preset_name

    def _merged_preset_state(self) -> PresetPack:
        base = get_preset(self._preset_name)
        functional = set(base.functional_predicates) | set(self.extra_functional_predicates)

        groups: list[set[str]] = [set(g) for g in base.exclusive_predicate_groups]
        for extra_group in self.extra_exclusive_groups:
            if len(extra_group) >= 2 and not any(g == extra_group for g in groups):
                groups.append(set(extra_group))

        aliases = dict(base.predicate_aliases)
        vocabulary = set(base.predicate_vocabulary) | set(functional) | set(self.extra_vocabulary)
        for grp in groups:
            vocabulary.update(grp)

        return PresetPack(
            name=self._preset_name,
            functional_predicates=functional,
            exclusive_predicate_groups=groups,
            predicate_aliases=aliases,
            predicate_vocabulary=vocabulary,
        )

    def get_active_rules(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of active memory rules."""
        state = self._merged_preset_state()
        return {
            "preset": state.name,
            "available_presets": list_presets(),
            "functional_predicates": sorted(state.functional_predicates),
            "exclusive_predicate_groups": [
                sorted(g)
                for g in sorted(state.exclusive_predicate_groups, key=lambda s: sorted(s))
            ],
            "predicate_aliases": dict(sorted(state.predicate_aliases.items())),
            "predicate_vocabulary": sorted(state.predicate_vocabulary),
        }

    def _apply_state_to_rag(self, rag: GraphRAG, state: PresetPack) -> None:
        if hasattr(rag.config, "functional_predicates"):
            rag.config.functional_predicates = set(state.functional_predicates)
        rag.config.exclusive_predicate_groups = [
            set(g) for g in state.exclusive_predicate_groups
        ]
        rag.config.predicate_aliases = dict(state.predicate_aliases)
        rag.config.predicate_vocabulary = sorted(state.predicate_vocabulary)

        if hasattr(rag, "set_functional_predicates"):
            rag.set_functional_predicates(set(state.functional_predicates))
        if hasattr(rag.extractor, "functional_predicates"):
            rag.extractor.functional_predicates = set(state.functional_predicates)
        rag.extractor.predicate_aliases = dict(state.predicate_aliases)
        rag.extractor.predicate_vocabulary = sorted(state.predicate_vocabulary)

    def extend_rules(
        self,
        preset: str | None = None,
        add_functional: list[str] | None = None,
        add_exclusive_groups: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        """Switch base preset and/or extend functional and exclusive predicate rules.

        Updates both ``EngineManager`` state for future realms and every currently
        cached :class:`GraphRAG` instance (including ``rag.extractor``).
        """
        if preset is not None and preset.strip():
            validated = get_preset(preset.strip())
            self._preset_name = validated.name

        if add_functional:
            for raw_pred in add_functional:
                norm_pred = normalise_predicate(raw_pred)
                if norm_pred:
                    self.extra_functional_predicates.add(norm_pred)
                    self.extra_vocabulary.add(norm_pred)

        if add_exclusive_groups:
            for raw_group in add_exclusive_groups:
                norm_group = {
                    np for item in (raw_group or []) if (np := normalise_predicate(item))
                }
                if len(norm_group) >= 2:
                    if not any(existing == norm_group for existing in self.extra_exclusive_groups):
                        self.extra_exclusive_groups.append(norm_group)
                    self.extra_vocabulary.update(norm_group)

        state = self._merged_preset_state()
        for rag in self._engines.values():
            self._apply_state_to_rag(rag, state)

        return self.get_active_rules()

    async def get_rag(self, realm: str | None = None) -> GraphRAG:
        """Return an initialized :class:`GraphRAG` engine for *realm*, caching per realm."""
        target_realm = (realm or self.cfg.realm).strip() or self.cfg.realm
        cached = self._engines.get(target_realm)
        if cached is not None:
            return cached

        async with self._lock:
            cached = self._engines.get(target_realm)
            if cached is not None:
                return cached

            db_uri = self.cfg.postgres_uri or await self._embedded.ensure_uri()
            state = self._merged_preset_state()

            rag_kwargs: dict[str, Any] = {
                "db_uri": db_uri,
                "realm": target_realm,
                "space": self.cfg.default_space,
                "schema_per_realm": self.cfg.schema_per_realm,
                "pool_min_size": self.cfg.pool_min_size,
                "pool_max_size": self.cfg.pool_max_size,
                "statement_cache_size": self.cfg.statement_cache_size,
                "embedding_model": self.cfg.embedding_model,
                "embedding_dim": self.cfg.embedding_dim,
                "embed_relations": True,
                "lexical_search": True,
                "record_retrieval_events": True,
                "auto_decompose": self.cfg.autonomous_llm_enabled,
                "contradiction_detection": self.cfg.contradiction_detection,
                "exclusive_predicate_groups": [
                    set(g) for g in state.exclusive_predicate_groups
                ],
                "predicate_aliases": dict(state.predicate_aliases),
                "predicate_vocabulary": sorted(state.predicate_vocabulary),
            }
            cfg_params = inspect.signature(RAGConfig.__init__).parameters
            if "functional_predicates" in cfg_params:
                rag_kwargs["functional_predicates"] = set(state.functional_predicates)
            if "enforce_embedding_model_match" in cfg_params:
                rag_kwargs["enforce_embedding_model_match"] = True

            rag_cfg = RAGConfig(**rag_kwargs)
            llm_service = self._llm_service_override or build_llm_service(self.cfg, rag_cfg)
            rag = GraphRAG(rag_cfg, llm=llm_service)
            self._apply_state_to_rag(rag, state)

            try:
                await rag.initialize()
            except Exception:
                try:
                    await rag.close()
                except Exception:
                    pass
                raise

            self._engines[target_realm] = rag
            return rag

    async def close(self) -> None:
        """Close all cached :class:`GraphRAG` instances and release embedded PostgreSQL state."""
        async with self._lock:
            engines = list(self._engines.values())
            self._engines.clear()

        first_err: Exception | None = None
        for rag in engines:
            try:
                await rag.close()
            except Exception as exc:
                if first_err is None:
                    first_err = exc

        await self._embedded.close()
        if first_err is not None:
            raise first_err
