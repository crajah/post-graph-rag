"""Built-in supersession and vocabulary preset packs for post-graph-rag-mcp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from post_graph_rag.extractor import normalise_predicate


@dataclass
class PresetPack:
    """Normalized declarative memory rules for a domain preset."""

    name: str
    functional_predicates: set[str]
    exclusive_predicate_groups: list[set[str]]
    predicate_aliases: dict[str, str]
    predicate_vocabulary: set[str]


def _build_preset(
    *,
    name: str,
    functional: Iterable[str],
    exclusive_groups: Iterable[Iterable[str]],
    aliases: Mapping[str, str],
    extra_vocabulary: Iterable[str] = (),
) -> PresetPack:
    norm_functional = {
        np for p in functional if (np := normalise_predicate(p))
    }

    norm_groups: list[set[str]] = []
    for group in exclusive_groups:
        norm_group = {np for p in group if (np := normalise_predicate(p))}
        if len(norm_group) >= 2:
            norm_groups.append(norm_group)

    norm_aliases: dict[str, str] = {}
    for raw_key, raw_target in aliases.items():
        nk = normalise_predicate(raw_key)
        nv = normalise_predicate(raw_target)
        if nk and nv and nk != nv:
            norm_aliases[nk] = nv

    vocab: set[str] = set(norm_functional)
    for grp in norm_groups:
        vocab.update(grp)
    vocab.update(norm_aliases.values())
    for item in extra_vocabulary:
        if np := normalise_predicate(item):
            vocab.add(np)

    return PresetPack(
        name=name,
        functional_predicates=norm_functional,
        exclusive_predicate_groups=norm_groups,
        predicate_aliases=norm_aliases,
        predicate_vocabulary=vocab,
    )


_ENGINEERING_PRESET = _build_preset(
    name="engineering",
    functional=(
        "uses_auth",
        "uses_database",
        "uses_framework",
        "uses_orm",
        "uses_queue",
        "uses_package_manager",
        "uses_test_runner",
        "uses_bundler",
        "uses_linter",
        "uses_styling",
        "deployed_on",
        "owned_by",
        "current_status",
        "default_model",
        "primary_branch",
        "api_version",
        "port_is",
        "located_at",
        "runtime_version",
    ),
    exclusive_groups=(
        {"enabled_in", "disabled_in"},
        {"depends_on", "removed_dependency_on"},
        {"approved", "deprecated", "rejected"},
        {"active", "archived"},
    ),
    aliases={
        "auth_provider": "uses_auth",
        "uses_auth_provider": "uses_auth",
        "auth_library": "uses_auth",
        "database_driver": "uses_database",
        "db_engine": "uses_database",
        "uses_db": "uses_database",
        "hosted_on": "deployed_on",
        "runs_on": "deployed_on",
        "job_queue": "uses_queue",
        "background_jobs": "uses_queue",
        "package_manager": "uses_package_manager",
        "css_framework": "uses_styling",
        "styling_framework": "uses_styling",
        "listen_port": "port_is",
        "listening_port": "port_is",
        "test_runner": "uses_test_runner",
        "test_framework": "uses_test_runner",
        "default_branch": "primary_branch",
        "main_branch": "primary_branch",
    },
)

_GENERAL_MEMORY_PRESET = _build_preset(
    name="general_memory",
    functional=(
        "lives_in",
        "works_at",
        "role",
        "cfo",
        "ceo",
        "reports_to",
        "prefers",
        "current_stage",
        "headquartered_in",
    ),
    exclusive_groups=(
        {"employs", "formerly_employed", "terminated"},
        {"married_to", "divorced_from"},
        {"allied_with", "rival_of"},
        {"owns", "sold"},
    ),
    aliases={
        "has_role": "role",
        "job_title": "role",
        "serves_as_cfo_of": "cfo",
        "has_cfo": "cfo",
        "chief_financial_officer": "cfo",
        "serves_as_ceo_of": "ceo",
        "has_ceo": "ceo",
        "chief_executive_officer": "ceo",
        "resides_in": "lives_in",
        "employed_by": "works_at",
    },
)

_PRESET_REGISTRY: dict[str, PresetPack] = {
    _ENGINEERING_PRESET.name: _ENGINEERING_PRESET,
    _GENERAL_MEMORY_PRESET.name: _GENERAL_MEMORY_PRESET,
}


def list_presets() -> list[str]:
    """Return available built-in preset names in deterministic order."""
    return sorted(_PRESET_REGISTRY.keys())


def get_preset(name: str) -> PresetPack:
    """Return a copy of the named :class:`PresetPack`, or raise :class:`ValueError`."""
    key = (name or "").strip().lower()
    pack = _PRESET_REGISTRY.get(key)
    if pack is None:
        available = ", ".join(list_presets())
        raise ValueError(f"Unknown memory preset {name!r}. Available presets: {available}.")

    return PresetPack(
        name=pack.name,
        functional_predicates=set(pack.functional_predicates),
        exclusive_predicate_groups=[set(grp) for grp in pack.exclusive_predicate_groups],
        predicate_aliases=dict(pack.predicate_aliases),
        predicate_vocabulary=set(pack.predicate_vocabulary),
    )
