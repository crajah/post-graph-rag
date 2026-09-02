"""Corpus deltas: what changed since an instant, answered from belief time.

Every relation carries ``t_created`` and, once superseded or withdrawn,
``t_expired``; entities carry ``dormant_since`` and documents an ``indexed_at``
stamp is not needed because chunk rows carry ``created_at``. A consumer that
re-explores a corpus therefore never needs to rescan it: one summary poll says
whether anything moved, and one detail call says exactly what.

Predicates run server-side through post-graph 1.3.0 range queries. The stamps
are ISO-8601 UTC strings, and the library's cast rule -- str compares as text
-- is what makes ``("t_created", ">", T)`` an indexed range scan rather than a
fetch-and-filter.

Two boundaries are deliberate. Relations that predate the belief-time fields
carry no ``t_created`` and never appear in a delta: absence records that
nothing was stated, the same principle as an absent validity period. And
communities are rebuilt wholesale rather than edited, so they report a
staleness flag, not a fake diff.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from post_graph import RESERVED_SPACE_ALL

_INCLUDE = ("relations", "entities", "documents", "communities")


@dataclass
class CorpusDelta:
    """Everything that changed after ``since``, with a watermark for the next poll."""
    since: str
    as_of: str                       # database clock; pass as `since` next time
    counts: Dict[str, int] = field(default_factory=dict)
    new_relations: List[Any] = field(default_factory=list)
    superseded_relations: List[Any] = field(default_factory=list)
    new_entities: List[Any] = field(default_factory=list)
    dormant_entities: List[Any] = field(default_factory=list)
    revived_entities: List[Any] = field(default_factory=list)
    new_documents: List[Any] = field(default_factory=list)
    communities_stale: bool = False

    @property
    def empty(self) -> bool:
        return not any(self.counts.values())


class DeltaReader:
    def __init__(self, store):
        self.store = store            # RAGGraphStore
        self.client = store.client
        self.realm = store.realm

    async def _db_now(self) -> str:
        """The watermark comes from the database clock, not the caller's.

        A poller stamping its own clock can miss a write that committed in the
        skew between the two; the server's now() cannot.
        """
        async def _op(conn):
            return await conn.fetchval("SELECT now()")
        import asyncpg
        if isinstance(self.client.connection, asyncpg.Pool):
            async with self.client.connection.acquire() as conn:
                ts = await _op(conn)
        else:
            ts = await _op(self.client.connection)
        return ts.astimezone(timezone.utc).isoformat()

    async def changes_since(
        self,
        since: str,
        space: Optional[str] = None,
        include: Tuple[str, ...] = _INCLUDE,
        limit: int = 500,
        summary: bool = False,
    ) -> CorpusDelta:
        for name in include:
            if name not in _INCLUDE:
                raise ValueError(f"Unknown include {name!r}; allowed: {_INCLUDE}")
        space = space or self.store.space
        as_of = await self._db_now()
        delta = CorpusDelta(since=since, as_of=as_of)
        c, realm = self.client, self.realm

        if "relations" in include:
            new_w = [("t_created", ">", since)]
            sup_w = [("t_expired", ">", since)]
            delta.counts["new_relations"] = await c.count_edges(
                "relations", realm=realm, space=space, where=new_w)
            delta.counts["superseded_relations"] = await c.count_edges(
                "relations", realm=realm, space=space, where=sup_w)
            if not summary:
                delta.new_relations = await c.find_edges(
                    "relations", realm=realm, space=space, where=new_w,
                    order_by="t_created", limit=limit)
                delta.superseded_relations = await c.find_edges(
                    "relations", realm=realm, space=space, where=sup_w,
                    order_by="t_expired", limit=limit)

        if "entities" in include:
            # Entities have no belief stamp of their own; creation is the row
            # clock, dormancy transitions are the payload stamp.
            dor_w = [("dormant_since", ">", since)]
            delta.counts["dormant_entities"] = await c.count_vertices(
                "entities", realm=realm, space=space, where=dor_w)
            if not summary:
                delta.dormant_entities = await c.find_vertices(
                    "entities", realm=realm, space=space, where=dor_w,
                    order_by="dormant_since", limit=limit)
            new_e, revived = await self.store.entities_changed_since(
                since, space=space, limit=limit, count_only=summary)
            if summary:
                delta.counts["new_entities"], delta.counts["revived_entities"] = new_e, revived
            else:
                delta.new_entities, delta.revived_entities = new_e, revived
                delta.counts["new_entities"] = len(new_e)
                delta.counts["revived_entities"] = len(revived)

        if "documents" in include:
            docs, n = await self.store.documents_created_since(
                since, space=space, limit=limit, count_only=summary)
            delta.counts["new_documents"] = n
            if not summary:
                delta.new_documents = docs

        if "communities" in include:
            oldest = await self.store.oldest_community_build(space=space)
            newest = await self.store.latest_graph_write(space=space)
            delta.communities_stale = bool(oldest and newest and oldest < newest)
            delta.counts["communities_stale"] = int(delta.communities_stale)

        return delta
