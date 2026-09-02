"""Graph Store implementation wrapping post-graph and pgvector."""
import asyncio
import json
import re
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from post_graph import AsyncPostGraph, Vertex, Edge, RESERVED_SPACE_ALL
from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import SchemaError
from post_graph_rag.models import DocumentMetadata

logger = logging.getLogger(__name__)

VERTEX_TABLES = ("documents", "entities", "communities")


def _utc_now() -> str:
    """Transaction-time stamp: when this system came to believe something.

    UTC and ISO-8601 so it sorts lexically, which is what the as-of-belief
    filter compares against without parsing.
    """
    return datetime.now(timezone.utc).isoformat()


class RAGGraphStore:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = AsyncPostGraph(
            dsn=config.db_uri, schema_per_realm=config.schema_per_realm,
            pool_min_size=config.pool_min_size, pool_max_size=config.pool_max_size)
        self.realm = config.realm
        self.space = config.space or "default"
        # Guards the lazily-built lexical index against concurrent first use.
        self._fts_index_ready = False
        self._fts_index_lock = asyncio.Lock()

    async def connect(self):
        await self.client.connect()

    async def close(self):
        await self.client.close()

    # ------------------------------------------------------------------ schema

    async def initialize_schema(self):
        """Create graph tables for documents, entities, and relations with vector support.

        Failures are raised, not logged. A vertex table created without a working
        ``vector`` extension has no embedding column, which makes every subsequent
        similarity search return nothing — a failure mode that is invisible at
        write time and produces irrelevant answers at read time.
        """
        for table in VERTEX_TABLES:
            try:
                await self.client.create_vertex_table(
                    table,
                    realm=self.realm,
                    vector_dim=self.config.embedding_dim
                )
            except Exception as e:
                raise SchemaError(
                    f"Failed to create vertex table '{table}' with vector_dim="
                    f"{self.config.embedding_dim}: {e}. Ensure the pgvector extension is "
                    f"available (CREATE EXTENSION IF NOT EXISTS vector;)."
                ) from e

        try:
            await self.client.create_edge_table(
                "relations",
                from_vertex_table="entities",
                to_vertex_table="entities",
                realm=self.realm,
                # Enabling this on a table created without it adds the column
                # and its index in place; rows written earlier keep a NULL
                # embedding until re-indexed, and edge search skips those.
                vector_dim=self.config.embedding_dim if self.config.embed_relations else None
            )
            await self.client.create_edge_table(
                "doc_mentions",
                from_vertex_table="documents",
                to_vertex_table="entities",
                realm=self.realm
            )
            await self.client.create_edge_table(
                "community_members",
                from_vertex_table="communities",
                to_vertex_table="entities",
                realm=self.realm
            )
            # Hierarchy edges: parent community -> child community. Deleting a
            # community cascades these away with it, so clear_communities needs
            # no special handling for levels.
            await self.client.create_edge_table(
                "community_children",
                from_vertex_table="communities",
                to_vertex_table="communities",
                realm=self.realm
            )
        except Exception as e:
            raise SchemaError(f"Failed to create edge tables: {e}") from e

        if self.config.record_retrieval_events:
            # Coverage telemetry table: plain rows, no vector column. Created
            # only when the flag is on, so realms of non-telemetry users gain
            # nothing.
            await self.client.create_vertex_table("retrieval_events", realm=self.realm)
            await self.client.create_payload_index("retrieval_events", realm=self.realm, key="ts")

        # Belief-time delta polls run range predicates over these payload
        # keys; the expression indexes make them index scans. Idempotent.
        for key in ("t_created", "t_expired"):
            await self.client.create_payload_index("relations", realm=self.realm, key=key)
        await self.client.create_payload_index("entities", realm=self.realm, key="dormant_since")
        # Hierarchy level: numeric expression index matching the numeric cast
        # the level predicate compiles to. Expression indexes apply to
        # existing tables, so realms created before the hierarchy gain it too.
        await self.client.create_payload_index("communities", realm=self.realm,
                                               key="level", numeric=True)

        await self._verify_vector_columns()
        await self._ensure_entity_name_index()

    async def _verify_vector_columns(self):
        """Assert every vertex table has an embedding column of the configured width.

        Catches the case where a table was previously created with a different
        ``embedding_dim`` (or before pgvector was installed) and would otherwise
        silently accept writes while failing every search.
        """
        for table in VERTEX_TABLES:
            table_ref = self.client._get_table_ref(table, self.realm)
            rows = await self.client._fetch(
                "SELECT format_type(a.atttypid, a.atttypmod) AS coltype "
                "FROM pg_attribute a "
                "WHERE a.attrelid = $1::regclass AND a.attname = 'embedding' "
                "AND NOT a.attisdropped",
                table_ref
            )
            if not rows:
                raise SchemaError(
                    f"Table '{table}' has no 'embedding' column. Vector search cannot work. "
                    f"This usually means the pgvector extension was unavailable when the "
                    f"table was created. Install pgvector, then drop and recreate the table."
                )
            coltype = rows[0]["coltype"]
            expected = f"vector({self.config.embedding_dim})"
            if coltype != expected:
                raise SchemaError(
                    f"Table '{table}' has embedding column of type '{coltype}' but "
                    f"RAGConfig.embedding_dim implies '{expected}'. Writes would fail or "
                    f"searches would return nothing. Align embedding_dim with the existing "
                    f"table, or recreate the table."
                )

    async def _ensure_entity_name_index(self):
        """Unique index backing entity resolution by canonical name.

        Entities are unique per (realm, space, lower(name)); this both enforces
        the invariant and makes the name lookup in :meth:`upsert_entity` an index hit.
        """
        table_ref = self.client._get_table_ref("entities", self.realm)
        index_name = f"{self.realm}_entities_name_uniq"
        try:
            await self.client._execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON {table_ref} '
                f"(realm, space, lower(payload->>'name'))"
            )
        except Exception as e:
            raise SchemaError(
                f"Failed to create entity name uniqueness index: {e}. Existing duplicate "
                f"entity rows must be merged before entity resolution can be enforced."
            ) from e

    # --------------------------------------------------------------- documents

    async def add_document(self, text: str, embedding: List[float], metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None, space: Optional[str] = None) -> Vertex:
        """Insert a document text chunk with embedding and structured metadata."""
        meta_dict = {}
        if isinstance(metadata, DocumentMetadata):
            meta_dict = metadata.to_dict()
        elif isinstance(metadata, dict):
            meta_dict = DocumentMetadata.from_dict(metadata).to_dict()

        target_space = space or meta_dict.get("space") or self.space
        payload = {"text": text, **meta_dict}
        return await self.client.add_vertex(
            "documents",
            realm=self.realm,
            space=target_space,
            payload=payload,
            embedding=embedding
        )

    # ------------------------------------------------------- document identity

    async def find_document_chunks(self, doc_key: str, space: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch the stored chunks for a document key, with their content hashes."""
        target_space = space or self.space
        table_ref = self.client._get_table_ref("documents", self.realm)
        rows = await self.client._fetch(
            f"SELECT id, payload FROM {table_ref} "
            f"WHERE realm = $1 AND space = $2 AND payload->>'doc_key' = $3 "
            f"ORDER BY (payload->>'paragraph')::int NULLS LAST, id",
            self.realm, target_space, doc_key,
        )
        out = []
        for r in rows:
            payload = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
            out.append({"id": str(r["id"]), "content_hash": payload.get("content_hash"), "payload": payload})
        return out

    async def delete_document_chunks(self, doc_key: str, space: Optional[str] = None) -> Dict[str, int]:
        """Remove a document's chunks and their mention edges.

        Entities are never deleted here. One whose last mention disappears is
        marked dormant instead, so the graph keeps what it learned while no
        longer presenting it as current.
        """
        target_space = space or self.space
        chunks = await self.find_document_chunks(doc_key, space=target_space)
        if not chunks:
            return {"chunks": 0, "mentions": 0, "dormant": 0}

        chunk_ids = [int(c["id"]) for c in chunks]
        mentions_ref = self.client._get_table_ref("doc_mentions", self.realm)
        docs_ref = self.client._get_table_ref("documents", self.realm)

        touched = await self.client._fetch(
            f"SELECT DISTINCT to_id FROM {mentions_ref} WHERE realm = $1 AND from_id = ANY($2::bigint[])",
            self.realm, chunk_ids,
        )
        removed = await self.client._fetch(
            f"DELETE FROM {mentions_ref} WHERE realm = $1 AND from_id = ANY($2::bigint[]) RETURNING id",
            self.realm, chunk_ids,
        )
        await self.client._execute(
            f"DELETE FROM {docs_ref} WHERE realm = $1 AND id = ANY($2::bigint[])",
            self.realm, chunk_ids,
        )

        withdrawn = await self._withdraw_relation_sources(chunk_ids, space=target_space)
        dormant = await self.refresh_dormancy(
            [str(t["to_id"]) for t in touched], space=target_space
        )
        return {"chunks": len(chunk_ids), "mentions": len(removed),
                "dormant": dormant, "relations_withdrawn": withdrawn}

    async def _withdraw_relation_sources(self, chunk_ids: List[int], space: Optional[str] = None) -> int:
        """Remove deleted chunks from relation provenance and recompute weight.

        A relation left with no contributing chunk is marked dormant rather than
        deleted, matching how orphaned entities are handled: the assertion was
        genuinely made once, and the audit trail is meant to keep it.
        """
        target_space = space or self.space
        rels_ref = self.client._get_table_ref("relations", self.realm)
        wanted = {str(c) for c in chunk_ids}
        rows = await self.client._fetch(
            f"SELECT id, payload FROM {rels_ref} WHERE realm = $1 AND space = $2",
            self.realm, target_space,
        )
        touched = 0
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            sources = [s for s in (payload.get("sources") or [])]
            remaining = [s for s in sources if s not in wanted]
            if len(remaining) == len(sources):
                continue
            payload["sources"] = remaining
            payload["weight"] = max(1, len(remaining))
            if remaining:
                if payload.pop("dormant_since", None) is not None:
                    # A revival is an event, not just an absence: stamping it is
                    # what lets changes_since() report it from belief time.
                    payload["revived_at"] = datetime.now(timezone.utc).isoformat()
            else:
                payload["dormant_since"] = datetime.now(timezone.utc).isoformat()
            await self.client._execute(
                f"UPDATE {rels_ref} SET payload = $1::jsonb WHERE realm = $2 AND id = $3",
                json.dumps(payload), self.realm, int(row["id"]),
            )
            touched += 1
        return touched

    async def refresh_dormancy(self, entity_ids: List[str], space: Optional[str] = None) -> int:
        """Mark entities with no remaining mentions dormant, and revive the rest.

        Dormancy tracks *document mentions* only. It is deliberately independent
        of relation supersession: a superseded relation still evidences that its
        entities exist, so it must never push an entity dormant.
        """
        if not entity_ids:
            return 0
        target_space = space or self.space
        ents_ref = self.client._get_table_ref("entities", self.realm)
        mentions_ref = self.client._get_table_ref("doc_mentions", self.realm)

        rows = await self.client._fetch(
            f"SELECT e.id, e.payload, "
            f"       (SELECT count(*) FROM {mentions_ref} m "
            f"        WHERE m.realm = e.realm AND m.to_id = e.id) AS mentions "
            f"FROM {ents_ref} e WHERE e.realm = $1 AND e.space = $2 AND e.id = ANY($3::bigint[])",
            self.realm, target_space, [int(e) for e in entity_ids],
        )

        marked = 0
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            has_mentions = int(row["mentions"]) > 0
            was_dormant = payload.get("dormant_since") is not None

            if has_mentions and was_dormant:
                payload.pop("dormant_since", None)          # revived by a new mention
                payload["revived_at"] = datetime.now(timezone.utc).isoformat()
            elif not has_mentions and not was_dormant:
                payload["dormant_since"] = datetime.now(timezone.utc).isoformat()
                marked += 1
            else:
                continue

            await self.client._execute(
                f"UPDATE {ents_ref} SET payload = $1::jsonb WHERE realm = $2 AND id = $3",
                json.dumps(payload), self.realm, int(row["id"]),
            )
        return marked

    # ---------------------------------------------------------------- entities

    async def find_entity_by_name(self, name: str, space: Optional[str] = None) -> Optional[Vertex]:
        """Look up an entity by canonical name, then by recorded alias.

        Alias lookup is what merges 'Babbage' into 'Charles Babbage'. Without it
        the same real-world entity fragments across vertices, and a traversal
        starting at one form cannot reach facts attached to the other.
        """
        target_space = space or self.space
        table_ref = self.client._get_table_ref("entities", self.realm)

        rows = await self.client._fetch(
            f"SELECT id FROM {table_ref} "
            f"WHERE realm = $1 AND space = $2 AND lower(payload->>'name') = lower($3) LIMIT 1",
            self.realm, target_space, name
        )
        if not rows:
            rows = await self.client._fetch(
                f"SELECT id FROM {table_ref} "
                f"WHERE realm = $1 AND space = $2 "
                f"AND EXISTS (SELECT 1 FROM jsonb_array_elements_text("
                f"  COALESCE(payload->'aliases', '[]'::jsonb)) a WHERE lower(a) = lower($3)) "
                f"LIMIT 1",
                self.realm, target_space, name
            )
        if not rows:
            return None
        return await self.client.get_vertex("entities", realm=self.realm, vertex_id=str(rows[0]["id"]))

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float],
        space: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> Vertex:
        """Upsert an entity vertex, resolving by canonical name or alias.

        Without this, every mention of an entity creates a fresh vertex, so the same
        real-world entity appearing in two documents ends up as two disconnected
        nodes — which defeats cross-document graph traversal.

        Merge policy: a more specific value never loses to a placeholder. The
        engine creates bare ``Concept`` stubs for triple endpoints that were not
        returned as full entities, and those must not overwrite a real type or
        description recorded earlier. The longer of two competing canonical names
        wins, so 'Babbage' upgrades to 'Charles Babbage' rather than the reverse.
        """
        target_space = space or self.space
        if target_space == RESERVED_SPACE_ALL:
            raise ValueError(
                f"'{RESERVED_SPACE_ALL}' is a query-time filter only and cannot be written to."
            )
        incoming_aliases = [a for a in (aliases or []) if a and a.strip()]

        existing = await self.find_entity_by_name(name, space=target_space)
        if existing is None:
            for alias in incoming_aliases:
                existing = await self.find_entity_by_name(alias, space=target_space)
                if existing is not None:
                    break

        if existing is None:
            payload = {
                "name": name,
                "type": entity_type,
                "description": description,
                "aliases": sorted({a for a in incoming_aliases if a.lower() != name.lower()}),
            }
            try:
                return await self.client.upsert_vertex(
                    "entities",
                    realm=self.realm,
                    space=target_space,
                    payload=payload,
                    embedding=embedding
                )
            except Exception:
                # Lost a race against a concurrent writer inserting the same name.
                existing = await self.find_entity_by_name(name, space=target_space)
                if existing is None:
                    raise

        prev = existing.payload or {}
        prev_name = prev.get("name") or name
        # Prefer the fuller surface form as canonical; demote the other to alias.
        canonical = name if len(name) > len(prev_name) else prev_name
        superseded = prev_name if canonical != prev_name else name

        merged_aliases = {a for a in prev.get("aliases") or [] if a}
        merged_aliases.update(incoming_aliases)
        if superseded and superseded.lower() != canonical.lower():
            merged_aliases.add(superseded)
        merged_aliases = sorted({a for a in merged_aliases if a.lower() != canonical.lower()})

        merged_type = entity_type if entity_type and entity_type != "Concept" else prev.get("type") or entity_type
        merged_desc = prev.get("description") or description or ""
        if description and len(description) > len(merged_desc):
            merged_desc = description

        payload = {
            "name": canonical,
            "type": merged_type,
            "description": merged_desc,
            "aliases": merged_aliases,
        }

        return await self.client.upsert_vertex(
            "entities",
            realm=self.realm,
            vertex_id=existing.id,
            space=target_space,
            payload=payload,
            embedding=embedding
        )

    # ------------------------------------------------------------------- edges

    async def find_relation(self, from_id: str, relation_type: str, to_id: str, space: Optional[str] = None) -> Optional[Edge]:
        """Locate an existing edge for a (subject, predicate, object) triple."""
        target_space = space or self.space
        table_ref = self.client._get_table_ref("relations", self.realm)
        rows = await self.client._fetch(
            f"SELECT id FROM {table_ref} WHERE realm = $1 AND space = $2 "
            f"AND from_id = $3 AND to_id = $4 AND relation_type = $5 LIMIT 1",
            self.realm, target_space, int(from_id), int(to_id), relation_type
        )
        if not rows:
            return None
        return await self.client.get_edge("relations", realm=self.realm, edge_id=str(rows[0]["id"]))

    async def add_relation(
        self,
        from_entity: Vertex,
        to_entity: Vertex,
        relation_type: str,
        description: Optional[str] = None,
        space: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        negated: bool = False,
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        source_chunk: Optional[str] = None,
    ) -> Edge:
        """Upsert a relationship edge between two entity vertices.

        The same triple extracted from two chunks must not become two edges.
        Re-observing a relation instead increments ``weight``, which the engine
        already reads when ranking and rendering relations but which nothing
        previously wrote.

        ``embedding`` is only stored when ``RAGConfig.embed_relations`` was set
        at schema creation time, since that is what gives the edge table a
        vector column.
        """
        target_space = space or self.space
        existing = await self.find_relation(from_entity.id, relation_type, to_entity.id, space=target_space)

        if existing is not None:
            prev = existing.payload or {}
            # Weight counts the DISTINCT chunks that asserted this relation, not
            # the number of times it was written. Incrementing blindly meant
            # re-indexing a document made its claims look independently
            # corroborated.
            prior_sources = set(prev.get("sources") or [])
            if source_chunk:
                sources = sorted(prior_sources | {source_chunk})
                weight = max(1, len(sources))
            else:
                # No provenance supplied (direct store use): fall back to counting
                # observations, which is the only signal available.
                sources = sorted(prior_sources)
                weight = int(prev.get("weight", 1)) + 1
            payload = {
                "description": description or prev.get("description", ""),
                "sources": sources,
                "weight": weight,
                "negated": bool(prev.get("negated", False)) and negated,
                "confidence": max(float(prev.get("confidence", 0.0)), float(confidence)),
                # A stated period, once known, is kept; re-observing the relation
                # without a period must not erase it.
                "valid_from": valid_from or prev.get("valid_from"),
                "valid_to": valid_to or prev.get("valid_to"),
                # Re-observing a relation does not restart its transaction time:
                # the system has believed it since the first observation, and
                # resetting this would erase that from the record.
                "t_created": prev.get("t_created") or _utc_now(),
                "t_expired": prev.get("t_expired"),
            }
            if prev.get("superseded_by") is not None:
                payload["superseded_by"] = prev["superseded_by"]
            return await self.client.upsert_edge(
                "relations",
                realm=self.realm,
                edge_id=existing.id,
                space=target_space,
                from_id=from_entity.id,
                to_id=to_entity.id,
                relation_type=relation_type,
                payload=payload,
                check_cycle=False,
                embedding=embedding if self.config.embed_relations else None
            )

        payload = {
            "description": description or "",
            "sources": [source_chunk] if source_chunk else [],
            "weight": 1,
            "negated": bool(negated),
            "confidence": float(confidence),
            # Validity: when the fact was true in the world, as the text states.
            "valid_from": valid_from,
            "valid_to": valid_to,
            # Transaction time: when this system came to believe it. The two are
            # independent — a filing published in 2024 can assert something true
            # in 2019 — and collapsing them makes "what did we believe last
            # March" unanswerable, which is the question an audit asks.
            "t_created": _utc_now(),
            "t_expired": None,
        }
        return await self.client.add_edge(
            "relations",
            realm=self.realm,
            space=target_space,
            from_id=from_entity.id,
            to_id=to_entity.id,
            relation_type=relation_type,
            payload=payload,
            check_cycle=False,
            embedding=embedding if self.config.embed_relations else None
        )

    async def _ensure_relation_fts_index(self):
        """GIN index over the relation text, for the lexical channel.

        Built lazily on first use rather than at schema creation, so an existing
        deployment gains it without a migration and one that never calls the
        lexical channel never pays for it.

        `IF NOT EXISTS` is not atomic. Two sessions can both find the index
        absent and then race to insert into pg_class, and the loser gets a
        unique violation on pg_class_relname_nsp_index rather than the silent
        no-op the clause implies. Concurrent first queries hit this reliably —
        six parallel questions against a fresh realm failed here, and the
        failure surfaced as a lost query rather than as anything about indexes.

        Two guards, because they cover different races. The lock serialises
        callers inside this process; catching the violation covers separate
        processes and connections, which no in-process lock can reach. Either
        way the index exists afterwards, which is all the caller needs.
        """
        if self._fts_index_ready:
            return
        async with self._fts_index_lock:
            if self._fts_index_ready:
                return
            table_ref = self.client._get_table_ref("relations", self.realm)
            index_name = f"{self.realm}_relations_fts"
            try:
                await self.client._execute(
                    f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_ref} USING gin ('
                    f"to_tsvector('english', coalesce(relation_type, '') || ' ' || "
                    f"coalesce(payload->>'description', '')))"
                )
            except Exception as e:
                # 23505 is unique_violation; the message check covers drivers
                # that do not expose sqlstate. Anything else is a real failure
                # and must not be swallowed — a missing index would otherwise
                # turn into an empty lexical channel that looks disabled.
                text = f"{getattr(e, 'sqlstate', '')} {e}".lower()
                if "23505" not in text and "duplicate key" not in text:
                    raise
                logger.debug("Relation FTS index %r created concurrently.", index_name)
            self._fts_index_ready = True

    async def search_relations_text(self, query: str, top_k: int = 20,
                                    space: Optional[str] = None) -> List[Tuple[Edge, float]]:
        """Lexical search over relations, ranked by ts_rank.

        The third candidate generator, alongside entity traversal and relation
        embeddings. It exists because embeddings are weakest exactly where
        lexical matching is strongest: rare identifiers that carry the meaning
        but sit nowhere useful in vector space — a part number, a statute, a
        model designation like ``737-9``. Those are frequently the terms a
        question turns on, and a nearest-neighbour search will happily return
        semantically adjacent relations that mention none of them.

        Uses PostgreSQL's own full-text search, so this adds a GIN index and no
        new infrastructure.
        """
        if not (query or "").strip():
            return []
        await self._ensure_relation_fts_index()
        # OR the terms, do not AND them. websearch_to_tsquery and
        # plainto_tsquery both require every term to appear in the same row, so
        # a natural-language question demands that one relation contain all of
        # "role", "faa", "play", "across" and "filings" — which nothing does.
        # The channel then returns nothing and looks disabled rather than
        # wrong. Retrieval wants any-term matching, ranked; ts_rank already
        # rewards rows matching more of them.
        terms = [w for w in re.findall(r"[\w-]+", query.lower()) if len(w) > 2]
        if not terms:
            return []
        tsquery = " | ".join(terms)
        target_space = space or self.space
        table_ref = self.client._get_table_ref("relations", self.realm)
        rows = await self.client._fetch(
            f"SELECT r.id, r.realm, r.space, r.from_id, r.to_id, r.relation_type, r.payload, "
            f"       r.created_at, r.updated_at, "
            f"       ts_rank(to_tsvector('english', coalesce(r.relation_type, '') || ' ' || "
            f"               coalesce(r.payload->>'description', '')), "
            f"               to_tsquery('english', $3)) AS rank "
            f"FROM {table_ref} r "
            f"WHERE r.realm = $1 AND r.space = $2 "
            f"  AND to_tsvector('english', coalesce(r.relation_type, '') || ' ' || "
            f"      coalesce(r.payload->>'description', '')) "
            f"      @@ to_tsquery('english', $3) "
            f"ORDER BY rank DESC LIMIT $4",
            self.realm, target_space, tsquery, int(top_k),
        )
        out: List[Tuple[Edge, float]] = []
        for r in rows:
            out.append((self._row_to_edge(r), float(r["rank"])))
        return out

    async def search_similar_relations(self, query_vec: List[float], top_k: int = 5, space: Optional[str] = None) -> List[Tuple[Edge, float]]:
        """Vector similarity search over relation edges.

        Optional feature: requires ``RAGConfig.embed_relations``. Most retrieval
        reaches relations by traversing from a matched entity instead.
        """
        if not self.config.embed_relations:
            return []
        return await self.client.vector_search_edges(
            "relations", realm=self.realm, space=space, query_vector=query_vec, top_k=top_k
        )

    async def supersede_conflicting(
        self,
        from_id: str,
        to_id: str,
        new_predicate: str,
        new_edge_id: str,
        exclusive_groups: List[Set[str]],
        space: Optional[str] = None,
    ) -> List[str]:
        """Close earlier relations that the new one contradicts.

        Two predicates in the same exclusive group cannot both hold between the
        same pair — "friend_of" and "enemy_of" are not simultaneously true. Left
        alone, both edges reach retrieval as current facts and the answer becomes
        "they are friends and also enemies".

        Resolution is by document order, not by dates: the newer assertion wins.
        That deliberately avoids depending on the model to state a period, which
        it does only when the text happens to say so.

        Superseded edges are marked, never deleted — the history is the point.
        """
        predicate = (new_predicate or "").strip().lower()
        conflicts = {
            p for group in exclusive_groups
            if predicate in {g.lower() for g in group}
            for p in {g.lower() for g in group}
        } - {predicate}
        if not conflicts:
            return []

        target_space = space or self.space
        table_ref = self.client._get_table_ref("relations", self.realm)
        rows = await self.client._fetch(
            f"SELECT id, relation_type, payload FROM {table_ref} "
            f"WHERE realm = $1 AND space = $2 AND from_id = $3 AND to_id = $4 "
            f"AND lower(relation_type) = ANY($5::text[]) AND id <> $6",
            self.realm, target_space, int(from_id), int(to_id),
            sorted(conflicts), int(new_edge_id),
        )

        superseded = []
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            if payload.get("superseded_by") == str(new_edge_id):
                continue
            payload["superseded_by"] = str(new_edge_id)
            # The moment this system stopped believing the fact, as distinct
            # from the moment the fact stopped being true. A row that is
            # superseded is still the correct answer to a question asked about
            # an earlier belief state.
            payload["t_expired"] = _utc_now()
            await self.client._execute(
                f"UPDATE {table_ref} SET payload = $1::jsonb WHERE realm = $2 AND id = $3",
                json.dumps(payload), self.realm, int(row["id"]),
            )
            superseded.append(str(row["id"]))
        return superseded

    async def find_contradiction_candidates(
        self,
        from_id: str,
        new_edge_id: str,
        limit: int = 8,
        space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Current relations about the same subject, as candidates to contradict.

        Scoped to the subject rather than the subject-object pair, which is what
        `supersede_conflicting` uses. A contradiction usually changes the object
        — "lives in Paris" then "lives in Berlin" — so a pair-scoped search
        cannot see it by construction. That is precisely the case the
        declarative pass misses.

        Already-superseded relations are excluded: they have been retracted, and
        re-retracting them would rewrite `t_expired` and corrupt the belief
        history that supersession exists to preserve. Newest first, because a
        contradiction is nearly always with a recent assertion.
        """
        target_space = space or self.space
        table_ref = self.client._get_table_ref("relations", self.realm)
        rows = await self.client._fetch(
            f"SELECT id, relation_type, from_id, to_id, payload FROM {table_ref} "
            f"WHERE realm = $1 AND space = $2 AND from_id = $3 AND id <> $4 "
            f"AND (payload->>'superseded_by') IS NULL "
            f"ORDER BY id DESC LIMIT $5",
            self.realm, target_space, int(from_id), int(new_edge_id), int(limit),
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
            out.append({
                "id": str(row["id"]),
                "relation_type": row["relation_type"],
                "from_id": str(row["from_id"]),
                "to_id": str(row["to_id"]),
                "description": payload.get("description", ""),
                "valid_from": payload.get("valid_from"),
                "valid_to": payload.get("valid_to"),
            })
        return out

    async def mark_superseded(
        self,
        edge_ids: List[str],
        new_edge_id: str,
        space: Optional[str] = None,
    ) -> List[str]:
        """Retract specific relations in favour of a newer one.

        The write half of `supersede_conflicting`, split out so contradiction
        detection can reuse it and cannot drift from it. Both axes are set the
        same way: `superseded_by` records what replaced the fact, `t_expired`
        when this system stopped believing it.
        """
        if not edge_ids:
            return []
        table_ref = self.client._get_table_ref("relations", self.realm)
        superseded = []
        for edge_id in edge_ids:
            rows = await self.client._fetch(
                f"SELECT id, payload FROM {table_ref} WHERE realm = $1 AND id = $2",
                self.realm, int(edge_id),
            )
            if not rows:
                continue
            payload = rows[0]["payload"] if isinstance(rows[0]["payload"], dict) else json.loads(rows[0]["payload"])
            if payload.get("superseded_by"):
                continue
            payload["superseded_by"] = str(new_edge_id)
            payload["t_expired"] = _utc_now()
            await self.client._execute(
                f"UPDATE {table_ref} SET payload = $1::jsonb WHERE realm = $2 AND id = $3",
                json.dumps(payload), self.realm, int(edge_id),
            )
            superseded.append(str(edge_id))
        return superseded

    async def add_doc_mention(self, doc_vertex: Vertex, entity_vertex: Vertex, space: Optional[str] = None) -> Optional[Edge]:
        """Link a document chunk to an entity it mentions, at most once.

        Populates the ``doc_mentions`` edge table, which the schema created but
        nothing previously wrote to.
        """
        target_space = space or self.space
        table_ref = self.client._get_table_ref("doc_mentions", self.realm)
        rows = await self.client._fetch(
            f"SELECT 1 FROM {table_ref} WHERE realm = $1 AND space = $2 "
            f"AND from_id = $3 AND to_id = $4 LIMIT 1",
            self.realm, target_space, int(doc_vertex.id), int(entity_vertex.id)
        )
        if rows:
            return None
        return await self.client.add_edge(
            "doc_mentions",
            realm=self.realm,
            space=target_space,
            from_id=doc_vertex.id,
            to_id=entity_vertex.id,
            relation_type="mentions",
            payload={},
            check_cycle=False
        )

    async def find_chunks_mentioning(
        self,
        entity_ids: List[str],
        limit: int = 10,
        space: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Vertex]:
        """Fetch document chunks that mention any of the given entities.

        This is the payoff of ``doc_mentions``: a question whose wording matches
        an entity but not the passage that explains it still reaches the passage,
        including passages in other documents. Chunks mentioning more of the
        matched entities rank first.
        """
        if not entity_ids:
            return []
        target_space = space or self.space
        docs_ref = self.client._get_table_ref("documents", self.realm)
        mentions_ref = self.client._get_table_ref("doc_mentions", self.realm)

        args: List[Any] = [self.realm, [int(e) for e in entity_ids], limit]
        space_clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            space_clause = f" AND d.space = ${len(args)}"

        rows = await self.client._fetch(
            f"SELECT d.realm, d.id, d.space, d.fqid, d.payload, d.created_at, d.updated_at, "
            f"       count(*) AS hits "
            f"FROM {mentions_ref} m JOIN {docs_ref} d ON d.realm = m.realm AND d.id = m.from_id "
            f"WHERE m.realm = $1 AND m.to_id = ANY($2::bigint[]){space_clause} "
            f"GROUP BY d.realm, d.id, d.space, d.fqid, d.payload, d.created_at, d.updated_at "
            f"ORDER BY hits DESC, d.id ASC LIMIT $3",
            *args
        )

        excluded = {str(x) for x in (exclude_ids or [])}
        out = []
        for r in rows:
            if str(r["id"]) in excluded:
                continue
            out.append(Vertex(
                realm=r["realm"], id=str(r["id"]), space=r.get("space") or "default",
                fqid=r["fqid"],
                payload=r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"]),
                created_at=r["created_at"], updated_at=r["updated_at"],
                table_name="documents", _client=self.client,
            ))
        return out

    # ------------------------------------------------------------- communities

    async def graph_snapshot(self, space: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch the entity graph for clustering.

        Returns (entities, relations) as plain dicts. Negated relations are
        included but flagged, so a detector can weight them and a report can say
        the relation is denied rather than asserted.
        """
        target_space = space or self.space
        ents_ref = self.client._get_table_ref("entities", self.realm)
        rels_ref = self.client._get_table_ref("relations", self.realm)

        args: List[Any] = [self.realm]
        space_clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            space_clause = " AND space = $2"

        ent_rows = await self.client._fetch(
            f"SELECT id, payload FROM {ents_ref} WHERE realm = $1{space_clause}", *args
        )
        rel_rows = await self.client._fetch(
            f"SELECT id, from_id, to_id, relation_type, payload FROM {rels_ref} "
            f"WHERE realm = $1{space_clause}", *args
        )

        def payload(row):
            return row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])

        entities = []
        for r in ent_rows:
            p = payload(r)
            if self.config.exclude_dormant_entities and p.get("dormant_since"):
                continue
            entities.append({
                "id": str(r["id"]), "name": p.get("name"), "type": p.get("type"),
                "description": p.get("description"), "aliases": p.get("aliases") or [],
            })

        relations = []
        for r in rel_rows:
            p = payload(r)
            relations.append({
                "id": str(r["id"]), "from_id": str(r["from_id"]), "to_id": str(r["to_id"]),
                "predicate": r["relation_type"], "description": p.get("description", ""),
                "weight": p.get("weight", 1), "negated": bool(p.get("negated", False)),
                "superseded_by": p.get("superseded_by"),
                "valid_from": p.get("valid_from"), "valid_to": p.get("valid_to"),
                "t_created": p.get("t_created"), "t_expired": p.get("t_expired"),
            })
        return entities, relations

    async def clear_communities(self, space: Optional[str] = None) -> int:
        """Delete existing communities for a space.

        Communities are derived data: re-running detection replaces them rather
        than accumulating stale clusters alongside fresh ones.
        """
        target_space = space or self.space
        comm_ref = self.client._get_table_ref("communities", self.realm)
        args: List[Any] = [self.realm]
        space_clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            space_clause = " AND space = $2"
        rows = await self.client._fetch(
            f"DELETE FROM {comm_ref} WHERE realm = $1{space_clause} RETURNING id", *args
        )
        return len(rows)

    async def add_community(
        self,
        key: str,
        title: str,
        summary: str,
        embedding: List[float],
        level: int = 0,
        rating: float = 5.0,
        findings: Optional[List[Dict[str, Any]]] = None,
        entity_ids: Optional[List[str]] = None,
        space: Optional[str] = None,
    ) -> Vertex:
        """Store a community report as an embedded vertex, linked to its members."""
        target_space = space or self.space
        payload = {
            "key": key,
            "title": title,
            "summary": summary,
            "level": level,
            "rating": rating,
            "findings": findings or [],
            "size": len(entity_ids or []),
            # Communities are derived data. Recording when they were built lets
            # retrieval warn that a report predates the graph it describes.
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        vertex = await self.client.add_vertex(
            "communities",
            realm=self.realm,
            space=target_space,
            payload=payload,
            embedding=embedding,
        )
        for entity_id in entity_ids or []:
            await self.client.add_edge(
                "community_members",
                realm=self.realm,
                space=target_space,
                from_id=vertex.id,
                to_id=entity_id,
                relation_type="includes",
                payload={},
                check_cycle=False,
            )
        return vertex

    async def search_similar_communities(
        self, query_vec: List[float], top_k: int = 5, space: Optional[str] = None,
        level: Optional[int] = None,
    ) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over community reports.

        *level* filters inside the search rather than trimming its result, so
        a level-restricted top-k is a genuine top-k even when another level
        dominates the query's neighbourhood.
        """
        where = [("level", "=", int(level))] if level is not None else None
        return await self.client.vector_search(
            "communities", realm=self.realm, space=space, query_vector=query_vec,
            top_k=top_k, where=where
        )

    async def latest_graph_write(self, space: Optional[str] = None) -> Optional[str]:
        """Most recent write to entities or relations, for staleness checks."""
        target_space = space or self.space
        args: List[Any] = [self.realm]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = " AND space = $2"
        ents = self.client._get_table_ref("entities", self.realm)
        rels = self.client._get_table_ref("relations", self.realm)
        rows = await self.client._fetch(
            f"SELECT max(t) AS latest FROM ("
            f"  SELECT max(greatest(created_at, updated_at)) t FROM {ents} WHERE realm = $1{clause}"
            f"  UNION ALL"
            f"  SELECT max(greatest(created_at, updated_at)) t FROM {rels} WHERE realm = $1{clause}"
            f") x", *args
        )
        latest = rows[0]["latest"] if rows else None
        return latest.isoformat() if latest else None

    async def oldest_community_build(self, space: Optional[str] = None) -> Optional[str]:
        """Earliest ``built_at`` among stored community reports."""
        target_space = space or self.space
        comm = self.client._get_table_ref("communities", self.realm)
        args: List[Any] = [self.realm]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = " AND space = $2"
        rows = await self.client._fetch(
            f"SELECT min(payload->>'built_at') AS built FROM {comm} WHERE realm = $1{clause}", *args
        )
        return rows[0]["built"] if rows else None

    async def add_community_child(self, parent: Vertex, child: Vertex,
                                  space: Optional[str] = None) -> None:
        await self.client.add_edge(
            "community_children", realm=self.realm,
            from_id=parent.id, to_id=child.id,
            relation_type="has_child", space=space or self.space,
            payload={})

    async def communities_at_level(self, level: int, space: Optional[str] = None):
        """Community vertices at one hierarchy level, via an indexed predicate."""
        return await self.client.find_vertices(
            "communities", realm=self.realm, space=space or self.space,
            where=[("level", "=", level)])

    async def community_children(self, community_id: str,
                                 space: Optional[str] = None) -> List[str]:
        edges = await self.client.find_edges(
            "community_children", realm=self.realm, space=space or self.space,
            filters={}, relation_type="has_child")
        return [e.to_id for e in edges if str(e.from_id) == str(community_id)]

    async def record_retrieval_event(self, mode: str, query_text: str,
                                     entity_ids: List[str], community_ids: List[str],
                                     space: Optional[str] = None) -> None:
        """Best-effort coverage telemetry; the one declared exception to
        fail-loud. Read-side bookkeeping must never fail the query it
        describes, so failures are logged and swallowed."""
        try:
            import hashlib
            if not getattr(self, "_events_table_ready", False):
                # The flag can be turned on after a realm was initialised, and
                # a missing table would otherwise make every event write a
                # silent no-op forever -- the worst failure shape for
                # telemetry. Same lazy-create idiom as the lexical index.
                await self.client.create_vertex_table("retrieval_events", realm=self.realm)
                await self.client.create_payload_index("retrieval_events",
                                                       realm=self.realm, key="ts")
                self._events_table_ready = True
            await self.client.add_vertex(
                "retrieval_events", realm=self.realm, space=space or self.space,
                payload={
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                    "query_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
                    "entity_ids": [str(i) for i in entity_ids],
                    "community_ids": [str(i) for i in community_ids],
                })
        except Exception as e:                                   # noqa: BLE001
            logger.warning("retrieval event write failed (telemetry only): %s", e)

    async def coverage_stats(self, space: Optional[str] = None) -> List[Dict[str, Any]]:
        """Per-community member count, retrieval hits and last hit.

        Hits combine direct community retrieval with entity-level touches
        joined through community_members, so a community counts as explored
        when retrieval reached its members even if no report was returned.
        """
        target_space = space or self.space
        comm = self.client._get_table_ref("communities", self.realm)
        cm = self.client._get_table_ref("community_members", self.realm)
        ev = self.client._get_table_ref("retrieval_events", self.realm)
        args: List[Any] = [self.realm]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = f" AND c.space = ${len(args)}"
        rows = await self.client._fetch(f"""
            WITH ent_hits AS (
                SELECT cm.from_id AS community_id,
                       count(*) AS hits, max(e.payload->>'ts') AS last_hit
                FROM {ev} e
                JOIN LATERAL jsonb_array_elements_text(e.payload->'entity_ids') AS x(eid) ON true
                JOIN {cm} cm ON cm.realm = e.realm AND cm.to_id = x.eid::bigint
                WHERE e.realm = $1
                GROUP BY cm.from_id
            ), direct_hits AS (
                SELECT x.cid::bigint AS community_id,
                       count(*) AS hits, max(e.payload->>'ts') AS last_hit
                FROM {ev} e
                JOIN LATERAL jsonb_array_elements_text(e.payload->'community_ids') AS x(cid) ON true
                WHERE e.realm = $1
                GROUP BY x.cid::bigint
            ), members AS (
                SELECT cm.from_id AS community_id, count(*) AS members
                FROM {cm} cm WHERE cm.realm = $1 GROUP BY cm.from_id
            )
            SELECT c.id, c.payload->>'title' AS title,
                   COALESCE(m.members, 0) AS members,
                   COALESCE(eh.hits, 0) + COALESCE(dh.hits, 0) AS hits,
                   GREATEST(eh.last_hit, dh.last_hit) AS last_hit
            FROM {comm} c
            LEFT JOIN members m ON m.community_id = c.id
            LEFT JOIN ent_hits eh ON eh.community_id = c.id
            LEFT JOIN direct_hits dh ON dh.community_id = c.id
            WHERE c.realm = $1{clause}
            ORDER BY hits ASC, members DESC
        """, *args)
        return [{"community_id": str(r["id"]), "title": r["title"],
                 "members": int(r["members"]), "retrieval_hits": int(r["hits"]),
                 "last_hit_at": r["last_hit"]} for r in rows]

    async def dark_entities(self, space: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Active entities no retrieval event has ever touched: the frontier."""
        target_space = space or self.space
        ents = self.client._get_table_ref("entities", self.realm)
        ev = self.client._get_table_ref("retrieval_events", self.realm)
        args: List[Any] = [self.realm]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = f" AND en.space = ${len(args)}"
        args.append(limit)
        rows = await self.client._fetch(f"""
            SELECT en.id, en.payload->>'name' AS name
            FROM {ents} en
            WHERE en.realm = $1{clause}
              AND en.payload->>'dormant_since' IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM {ev} e
                JOIN LATERAL jsonb_array_elements_text(e.payload->'entity_ids') AS x(eid) ON true
                WHERE e.realm = en.realm AND x.eid::bigint = en.id)
            ORDER BY en.id ASC LIMIT ${len(args)}
        """, *args)
        return [{"entity_id": str(r["id"]), "name": r["name"]} for r in rows]

    async def entities_changed_since(self, since: str, space: Optional[str] = None,
                                     limit: int = 500, count_only: bool = False):
        """New entities (row creation) and revived entities (revived_at stamp)
        after *since*. Returns (new, revived) as row-dicts, or counts when
        *count_only*."""
        target_space = space or self.space
        ents = self.client._get_table_ref("entities", self.realm)
        # asyncpg types $2 from the timestamptz comparison, so the row-clock
        # predicates need a datetime; the payload-stamp predicates compare the
        # ISO string as text and take `since` unchanged.
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        args: List[Any] = [self.realm, since_dt]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = f" AND space = ${len(args)}"
        if count_only:
            new_n = await self.client._fetch(
                f"SELECT count(*) AS n FROM {ents} WHERE realm = $1 AND created_at > $2::timestamptz{clause}", *args)
            rev_args = [self.realm, since] + args[2:]
            rev_n = await self.client._fetch(
                f"SELECT count(*) AS n FROM {ents} WHERE realm = $1 AND payload->>'revived_at' > $2{clause}", *rev_args)
            return int(new_n[0]["n"]), int(rev_n[0]["n"])
        args.append(limit)
        new_rows = await self.client._fetch(
            f"SELECT id, fqid, payload, created_at FROM {ents} "
            f"WHERE realm = $1 AND created_at > $2::timestamptz{clause} "
            f"ORDER BY created_at ASC LIMIT ${len(args)}", *args)
        rev_args = [self.realm, since] + args[2:]
        rev_rows = await self.client._fetch(
            f"SELECT id, fqid, payload, created_at FROM {ents} "
            f"WHERE realm = $1 AND payload->>'revived_at' > $2{clause} "
            f"ORDER BY payload->>'revived_at' ASC LIMIT ${len(rev_args)}", *rev_args)
        def rows(rs):
            out = []
            for r in rs:
                p = r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])
                out.append({"id": str(r["id"]), "fqid": r["fqid"], "payload": p,
                            "created_at": r["created_at"].isoformat()})
            return out
        return rows(new_rows), rows(rev_rows)

    async def documents_created_since(self, since: str, space: Optional[str] = None,
                                      limit: int = 500, count_only: bool = False):
        """Documents (grouped by doc_key) whose first chunk arrived after
        *since*. Returns (rows, total_count); rows empty when *count_only*."""
        target_space = space or self.space
        docs = self.client._get_table_ref("documents", self.realm)
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        args: List[Any] = [self.realm, since_dt]
        clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            clause = f" AND space = ${len(args)}"
        grouped = (
            f"SELECT payload->>'doc_key' AS doc_key, count(*) AS chunks, "
            f"       min(created_at) AS first_created "
            f"FROM {docs} WHERE realm = $1{clause} "
            f"GROUP BY payload->>'doc_key' "
            f"HAVING min(created_at) > $2::timestamptz")
        n_rows = await self.client._fetch(
            f"SELECT count(*) AS n FROM ({grouped}) g", *args)
        total = int(n_rows[0]["n"])
        if count_only:
            return [], total
        args.append(limit)
        rows = await self.client._fetch(
            grouped + f" ORDER BY min(created_at) ASC LIMIT ${len(args)}", *args)
        return [{"doc_key": r["doc_key"], "chunks": int(r["chunks"]),
                 "first_created": r["first_created"].isoformat()} for r in rows], total

    async def count_communities(self, space: Optional[str] = None) -> int:
        target_space = space or self.space
        comm_ref = self.client._get_table_ref("communities", self.realm)
        args: List[Any] = [self.realm]
        space_clause = ""
        if target_space and target_space != RESERVED_SPACE_ALL:
            args.append(target_space)
            space_clause = " AND space = $2"
        rows = await self.client._fetch(
            f"SELECT count(*) AS n FROM {comm_ref} WHERE realm = $1{space_clause}", *args
        )
        return int(rows[0]["n"]) if rows else 0

    # --------------------------------------------------------------- retrieval

    async def search_similar_entities(self, query_vec: List[float], top_k: int = 5, space: Optional[str] = None) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over entity vertices, optionally scoped by space.

        Dormant entities — those whose last mentioning document was removed — are
        dropped. pgvector cannot filter on a JSONB field, so this over-fetches and
        post-filters rather than pushing the predicate into the query.
        """
        if not self.config.exclude_dormant_entities:
            return await self.client.vector_search(
                "entities", realm=self.realm, space=space, query_vector=query_vec, top_k=top_k
            )
        hits = await self.client.vector_search(
            "entities", realm=self.realm, space=space, query_vector=query_vec, top_k=top_k * 3
        )
        live = [(v, d) for v, d in hits if not (v.payload or {}).get("dormant_since")]
        return live[:top_k]

    async def search_similar_documents(self, query_vec: List[float], top_k: int = 5, space: Optional[str] = None) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over document vertices, optionally scoped by space."""
        return await self.client.vector_search(
            "documents", realm=self.realm, space=space, query_vector=query_vec, top_k=top_k
        )

    async def get_relation_endpoints(self, edge: Edge) -> Tuple[Optional[Vertex], Optional[Vertex]]:
        """Resolve the source and target entity vertices of a relation edge."""
        from_v = await self.client.get_vertex("entities", realm=self.realm, vertex_id=edge.from_id)
        to_v = await self.client.get_vertex("entities", realm=self.realm, vertex_id=edge.to_id)
        return from_v, to_v

    async def get_neighbors(self, entity_id: str, space: Optional[str] = None) -> List[Tuple[Edge, Vertex]]:
        """Get 1-hop outward relationships and target entities, scoped by space.

        Space scoping matters here: without it, traversal from a matched entity
        walks into other tenants' spaces even though the vector search that found
        the entity was correctly scoped.
        """
        vertex = await self.client.get_vertex("entities", realm=self.realm, vertex_id=entity_id)
        if not vertex:
            return []
        steps = await vertex.outgoing("relations")
        target_space = space or self.space
        results = []
        for step in steps:
            if target_space and target_space != RESERVED_SPACE_ALL:
                if step.edge.space != target_space or step.neighbor_vertex.space != target_space:
                    continue
            results.append((step.edge, step.neighbor_vertex))
        return results

    async def get_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 1,
        space: Optional[str] = None,
        as_of: Optional[str] = None,
        include_superseded: bool = False,
        relation_types: Optional[List[str]] = None,
        max_edges: int = 200,
    ) -> List[Tuple[Edge, Vertex, Vertex, int]]:
        """Relations reachable within ``max_hops`` of an entity, with endpoints.

        One hop answers "what is said about X". Chain questions — how a programme
        came to drive cash burn — need the edges between X's neighbours too, and
        those are never adjacent to X.

        Filtering happens inside the walk rather than afterwards, so a path is
        never routed *through* a superseded or out-of-period edge to reach
        something that would otherwise look current. Filtering the result set
        instead would leave exactly those laundered paths behind.

        Fan-out is the danger: on a dense graph three hops from one entity
        reaches tens of thousands of edges, which is noise rather than context.
        ``max_edges`` caps that, preferring the closest hops.
        """
        if max_hops <= 1:
            source = await self.client.get_vertex("entities", realm=self.realm, vertex_id=entity_id)
            if source is None:
                return []
            return [
                (edge, source, target, 1)
                for edge, target in await self.get_neighbors(entity_id, space=space)
            ]

        target_space = space or self.space
        rows = await self.client.traverse(
            realm=self.realm,
            start_table="entities",
            start_id=str(entity_id),
            edge_tables=["relations"],
            max_depth=max_hops,
            direction="out",
            relation_types=relation_types,
            as_of=as_of,
            payload_null_keys=None if include_superseded else ["superseded_by"],
            space=None if target_space == RESERVED_SPACE_ALL else target_space,
        )

        # Closest hops first, so truncation drops the most tenuous connections.
        # The hop number travels with each edge: downstream ranking sorts on it,
        # and without it a distant edge indexed later would displace an adjacent
        # one purely on assertion time.
        edge_ids: List[str] = []
        hop_of: Dict[str, int] = {}
        for row in sorted(rows, key=lambda r: r["depth"]):
            for i, eid in enumerate(row.get("edge_ids") or [], start=1):
                if eid not in hop_of:
                    hop_of[eid] = i
                    edge_ids.append(eid)
            if len(edge_ids) >= max_edges:
                break
        edge_ids = edge_ids[:max_edges]
        loaded = await self.get_relations_by_ids(edge_ids)
        return [(e, s, t, hop_of.get(e.id, 1)) for e, s, t in loaded]

    async def get_relations_by_ids(self, edge_ids: List[str]) -> List[Tuple[Edge, Vertex, Vertex]]:
        """Load relations and both endpoint vertices in two queries.

        Traversal returns identifiers; synthesis needs names and descriptions.
        Fetching each edge and vertex individually would issue thousands of round
        trips on a wide neighbourhood.
        """
        if not edge_ids:
            return []
        rel_ref = self.client._get_table_ref("relations", self.realm)
        ent_ref = self.client._get_table_ref("entities", self.realm)
        rows = await self.client._fetch(
            f"SELECT * FROM {rel_ref} WHERE realm = $1 AND id = ANY($2::bigint[])",
            self.realm, [int(e) for e in edge_ids],
        )
        wanted = {r["from_id"] for r in rows} | {r["to_id"] for r in rows}
        if not wanted:
            return []
        ent_rows = await self.client._fetch(
            f"SELECT * FROM {ent_ref} WHERE realm = $1 AND id = ANY($2::bigint[])",
            self.realm, list(wanted),
        )
        vertices = {r["id"]: self._row_to_vertex(r) for r in ent_rows}
        out = []
        for r in rows:
            src, tgt = vertices.get(r["from_id"]), vertices.get(r["to_id"])
            if src and tgt:
                out.append((self._row_to_edge(r), src, tgt))
        return out

    @staticmethod
    def _row_to_vertex(row) -> Vertex:
        payload = row["payload"]
        return Vertex(
            id=str(row["id"]), realm=row["realm"], space=row["space"],
            payload=json.loads(payload) if isinstance(payload, str) else (payload or {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
            table_name="entities",
        )

    @staticmethod
    def _row_to_edge(row) -> Edge:
        # updated_at carries the assertion time that ranks contradicting facts
        # newest-first, so it must survive the round trip from SQL.
        payload = row["payload"]
        return Edge(
            id=str(row["id"]), realm=row["realm"], space=row["space"],
            from_id=str(row["from_id"]), to_id=str(row["to_id"]),
            relation_type=row["relation_type"],
            payload=json.loads(payload) if isinstance(payload, str) else (payload or {}),
            created_at=row["created_at"], updated_at=row["updated_at"],
            table_name="relations",
        )

    async def get_all_relations(self, limit: int = 50, space: Optional[str] = None) -> List[Tuple[Edge, Vertex, Vertex]]:
        """Fetch relations with their source and target entity vertices, scoped by space.

        Returns list of (edge, from_vertex, to_vertex) tuples.
        """
        target_space = space or self.space
        entities = await self.client.get_vertices(
            "entities",
            realm=self.realm,
            space=target_space if target_space != RESERVED_SPACE_ALL else RESERVED_SPACE_ALL
        )

        results = []
        for entity in entities:
            if len(results) >= limit:
                break
            for edge, neighbor in await self.get_neighbors(entity.id, space=target_space):
                if len(results) >= limit:
                    break
                results.append((edge, entity, neighbor))
        return results
