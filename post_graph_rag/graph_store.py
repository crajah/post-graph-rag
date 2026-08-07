"""Graph Store implementation wrapping post-graph and pgvector."""
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from post_graph import AsyncPostGraph, Vertex, Edge, RESERVED_SPACE_ALL
from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import SchemaError
from post_graph_rag.models import DocumentMetadata

logger = logging.getLogger(__name__)

VERTEX_TABLES = ("documents", "entities")


class RAGGraphStore:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = AsyncPostGraph(dsn=config.db_uri, schema_per_realm=config.schema_per_realm)
        self.realm = config.realm
        self.space = config.space or "default"

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
                # Optional: relations are normally reached by traversing from a
                # matched entity, not by similarity, so this stays off by default.
                vector_dim=self.config.embedding_dim if self.config.embed_relations else None
            )
            await self.client.create_edge_table(
                "doc_mentions",
                from_vertex_table="documents",
                to_vertex_table="entities",
                realm=self.realm
            )
        except Exception as e:
            raise SchemaError(f"Failed to create edge tables: {e}") from e

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
            payload = {
                "description": description or prev.get("description", ""),
                "weight": int(prev.get("weight", 1)) + 1,
                "negated": bool(prev.get("negated", False)) and negated,
                "confidence": max(float(prev.get("confidence", 0.0)), float(confidence)),
            }
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
            "weight": 1,
            "negated": bool(negated),
            "confidence": float(confidence),
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

    # --------------------------------------------------------------- retrieval

    async def search_similar_entities(self, query_vec: List[float], top_k: int = 5, space: Optional[str] = None) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over entity vertices, optionally scoped by space."""
        return await self.client.vector_search(
            "entities", realm=self.realm, space=space, query_vector=query_vec, top_k=top_k
        )

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
