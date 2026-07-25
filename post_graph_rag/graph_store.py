"""Graph Store implementation wrapping post-graph and pgvector."""
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from post_graph import AsyncPostGraph, Vertex, Edge, TableNotFoundError
from post_graph_rag.config import RAGConfig
from post_graph_rag.models import DocumentMetadata

logger = logging.getLogger(__name__)

class RAGGraphStore:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = AsyncPostGraph(dsn=config.db_uri)
        self.realm = config.realm

    async def connect(self):
        await self.client.connect()

    async def close(self):
        await self.client.close()

    async def initialize_schema(self):
        """Create graph tables for documents, entities, and relations with vector support."""
        # 1. Documents vertex table
        try:
            await self.client.create_vertex_table(
                "documents",
                realm=self.realm,
                vector_dim=self.config.embedding_dim
            )
        except Exception as e:
            logger.info(f"Documents table creation note: {e}")

        # 2. Entities vertex table
        try:
            await self.client.create_vertex_table(
                "entities",
                realm=self.realm,
                vector_dim=self.config.embedding_dim
            )
        except Exception as e:
            logger.info(f"Entities table creation note: {e}")

        # 3. Entity-to-Entity relationship edges
        try:
            await self.client.create_edge_table(
                "relations",
                from_vertex_table="entities",
                to_vertex_table="entities",
                realm=self.realm
            )
        except Exception as e:
            logger.info(f"Relations edge table note: {e}")

        # 4. Document-to-Entity mention edges
        try:
            await self.client.create_edge_table(
                "doc_mentions",
                from_vertex_table="documents",
                to_vertex_table="entities",
                realm=self.realm
            )
        except Exception as e:
            logger.info(f"Doc Mentions edge table note: {e}")

    async def add_document(self, text: str, embedding: List[float], metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None) -> Vertex:
        """Insert or upsert a document text chunk with embedding and structured metadata."""
        meta_dict = {}
        if isinstance(metadata, DocumentMetadata):
            meta_dict = metadata.to_dict()
        elif isinstance(metadata, dict):
            meta_dict = DocumentMetadata.from_dict(metadata).to_dict()

        payload = {"text": text, **meta_dict}
        return await self.client.add_vertex(
            "documents",
            realm=self.realm,
            payload=payload,
            embedding=embedding
        )

    async def upsert_entity(self, name: str, entity_type: str, description: str, embedding: List[float]) -> Vertex:
        """Upsert an entity vertex by name."""
        payload = {"name": name, "type": entity_type, "description": description}
        # Fetch existing by searching name in payload if present, or upsert
        return await self.client.upsert_vertex(
            "entities",
            realm=self.realm,
            payload=payload,
            embedding=embedding
        )

    async def add_relation(self, from_entity: Vertex, to_entity: Vertex, relation_type: str, description: Optional[str] = None) -> Edge:
        """Create a relationship edge between two entity vertices."""
        payload = {"description": description or ""}
        return await self.client.add_edge(
            "relations",
            realm=self.realm,
            from_id=from_entity.id,
            to_id=to_entity.id,
            relation_type=relation_type,
            payload=payload,
            check_cycle=False
        )

    async def search_similar_entities(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over entity vertices."""
        try:
            return await self.client.vector_search("entities", realm=self.realm, query_vector=query_vec, top_k=top_k)
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            return []

    async def search_similar_documents(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[Vertex, float]]:
        """Vector similarity search over document vertices."""
        try:
            return await self.client.vector_search("documents", realm=self.realm, query_vector=query_vec, top_k=top_k)
        except Exception as e:
            logger.warning(f"Document vector search failed: {e}")
            return []

    async def get_neighbors(self, entity_id: str) -> List[Tuple[Edge, Vertex]]:
        """Get 1-hop outward relationships and target entities from an entity."""
        vertex = await self.client.get_vertex("entities", realm=self.realm, vertex_id=entity_id)
        if not vertex:
            return []
        steps = await vertex.outgoing("relations")
        return [(step.edge, step.neighbor_vertex) for step in steps]
