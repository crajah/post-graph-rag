"""Community detection over the entity graph.

Corpus-level questions ("what are the main themes here?") cannot be answered by
retrieving individual passages, because no single passage contains the answer.
GraphRAG's approach is to cluster the entity graph, summarise each cluster, and
answer from those summaries. This module supplies the clustering half.

The default detector is label propagation, which needs no third-party
dependency. Leiden is used instead when ``igraph`` and ``leidenalg`` are
installed, since it produces better-balanced communities on larger graphs. Any
callable matching :class:`CommunityDetector` can be supplied instead.
"""
import logging
from collections import defaultdict
from typing import Dict, Hashable, List, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

# (source, target, weight)
Edge = Tuple[str, str, float]


class CommunityDetector(Protocol):
    """Partitions nodes into communities.

    Returns a mapping of node id -> community id. Nodes absent from the mapping
    are treated as unassigned and skipped.
    """

    def __call__(self, nodes: Sequence[str], edges: Sequence[Edge]) -> Dict[str, int]:  # pragma: no cover
        ...


def _adjacency(nodes: Sequence[str], edges: Sequence[Edge]) -> Dict[str, List[Tuple[str, float]]]:
    adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
    for src, dst, weight in edges:
        if src == dst or src not in adj or dst not in adj:
            continue
        adj[src].append((dst, weight))
        adj[dst].append((src, weight))
    return adj


def label_propagation(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    max_iterations: int = 20,
) -> Dict[str, int]:
    """Semi-synchronous label propagation.

    Each node repeatedly adopts the highest-weighted label among its neighbours.
    Nodes are visited in a fixed order and ties broken by the smallest label, so
    the partition is deterministic for a given input — a randomised variant would
    give a different graph on every indexing run.
    """
    adj = _adjacency(nodes, edges)
    labels: Dict[str, str] = {n: n for n in nodes}
    order = sorted(nodes)

    for _ in range(max_iterations):
        changed = False
        for node in order:
            neighbours = adj.get(node) or []
            if not neighbours:
                continue
            weights: Dict[str, float] = defaultdict(float)
            for neighbour, weight in neighbours:
                weights[labels[neighbour]] += weight
            # Highest weight wins; smallest label breaks ties deterministically.
            best = min(
                (lbl for lbl, w in weights.items() if w == max(weights.values())),
            )
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break

    # Compact arbitrary label strings into stable integer community ids.
    ordered_labels = sorted({labels[n] for n in nodes})
    index = {lbl: i for i, lbl in enumerate(ordered_labels)}
    return {n: index[labels[n]] for n in nodes}


def leiden(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    resolution: float = 1.0,
    seed: int = 7,
) -> Dict[str, int]:
    """Leiden clustering via igraph/leidenalg.

    Raises ImportError when the optional dependencies are absent; callers should
    fall back to :func:`label_propagation`.
    """
    import igraph  # noqa: F401
    import leidenalg

    index = {n: i for i, n in enumerate(nodes)}
    graph = igraph.Graph(n=len(nodes), directed=False)
    weights = []
    pairs = []
    for src, dst, weight in edges:
        if src == dst or src not in index or dst not in index:
            continue
        pairs.append((index[src], index[dst]))
        weights.append(weight)
    graph.add_edges(pairs)
    if weights:
        graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = partition.membership
    return {n: int(membership[i]) for n, i in index.items()}


def default_detector(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    resolution: float = 1.0,
) -> Dict[str, int]:
    """Leiden when available, otherwise label propagation."""
    try:
        return leiden(nodes, edges, resolution=resolution)
    except ImportError:
        logger.debug("leidenalg/igraph not installed; using label propagation.")
        return label_propagation(nodes, edges)
    except Exception as e:
        logger.warning("Leiden clustering failed (%s); using label propagation.", e)
        return label_propagation(nodes, edges)


def group_by_community(
    assignment: Dict[str, int],
    min_size: int = 2,
) -> Dict[int, List[str]]:
    """Invert a node->community mapping, dropping communities below ``min_size``.

    Singletons are dropped by default: a one-entity "community" produces a report
    that says nothing the entity's own description does not already say.
    """
    groups: Dict[int, List[str]] = defaultdict(list)
    for node, community in assignment.items():
        groups[community].append(node)
    return {c: sorted(members) for c, members in groups.items() if len(members) >= min_size}
