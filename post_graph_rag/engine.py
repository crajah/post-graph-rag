import json
import logging
from typing import List, Dict, Any, Optional, Union
from post_graph import Vertex
from post_graph_rag.config import RAGConfig
from post_graph_rag.models import DocumentMetadata
from post_graph_rag.llm import LLMService
from post_graph_rag.extractor import GraphExtractor, ExtractionResult
from post_graph_rag.graph_store import RAGGraphStore

logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.llm = LLMService(self.config)
        self.extractor = GraphExtractor(self.llm)
        self.store = RAGGraphStore(self.config)

    async def initialize(self):
        """Initialize database connection and schema."""
        await self.store.connect()
        await self.store.initialize_schema()

    async def close(self):
        """Close database connection."""
        await self.store.close()

    async def index_document(self, text: str, metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None) -> Dict[str, Any]:
        """Index a document: extract entities/triples, compute embeddings, and populate graph."""
        meta_obj = metadata if isinstance(metadata, DocumentMetadata) else DocumentMetadata.from_dict(metadata or {})
        
        # 1. Compute embedding for document chunk
        doc_emb = await self.llm.get_embedding(text)
        doc_vertex = await self.store.add_document(text, doc_emb, meta_obj)

        # 2. Extract entities and triples using LLM
        extraction: ExtractionResult = await self.extractor.extract_from_text(text)

        entity_vertex_map = {}
        # 3. Insert/Upsert entities with embeddings
        for entity in extraction.entities:
            entity_text = f"{entity.name} ({entity.type}): {entity.description}"
            entity_emb = await self.llm.get_embedding(entity_text)
            e_vertex = await self.store.upsert_entity(
                name=entity.name,
                entity_type=entity.type,
                description=entity.description,
                embedding=entity_emb
            )
            entity_vertex_map[entity.name.lower()] = e_vertex

        # 4. Insert relationship edges
        added_relations = []
        for triple in extraction.triples:
            subj_key = triple.subject.lower()
            obj_key = triple.object.lower()

            subj_vertex = entity_vertex_map.get(subj_key)
            obj_vertex = entity_vertex_map.get(obj_key)

            if not subj_vertex:
                s_emb = await self.llm.get_embedding(triple.subject)
                subj_vertex = await self.store.upsert_entity(triple.subject, "Concept", "", s_emb)
                entity_vertex_map[subj_key] = subj_vertex

            if not obj_vertex:
                o_emb = await self.llm.get_embedding(triple.object)
                obj_vertex = await self.store.upsert_entity(triple.object, "Concept", "", o_emb)
                entity_vertex_map[obj_key] = obj_vertex

            edge = await self.store.add_relation(subj_vertex, obj_vertex, triple.predicate, triple.description)
            added_relations.append(edge)

        return {
            "document_id": doc_vertex.id,
            "entities_extracted": len(extraction.entities),
            "triples_extracted": len(extraction.triples),
            "entities": [e.name for e in extraction.entities],
            "metadata": meta_obj.to_dict()
        }

    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer user query by retrieving vector context + graph context and synthesizing answer."""
        query_vec = await self.llm.get_embedding(question)

        # 1. Search vector similarity for entities and documents
        similar_entities = await self.store.search_similar_entities(query_vec, top_k=top_k)
        similar_docs = await self.store.search_similar_documents(query_vec, top_k=top_k)

        # Fallback to direct vertex queries if vector search returned empty (e.g. non-vector mode)
        if not similar_entities:
            try:
                table_ref = self.store.client._get_table_ref("entities", self.config.realm)
                rows = await self.store.client._fetch(f"SELECT realm, id, fqid, payload, created_at, updated_at FROM {table_ref} LIMIT $1", top_k * 3)
                for r in rows:
                    v = Vertex(realm=r['realm'], id=str(r['id']), fqid=r['fqid'], payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']), created_at=r['created_at'], updated_at=r['updated_at'], table_name="entities", _client=self.store.client)
                    similar_entities.append((v, 0.0))
            except Exception:
                pass

        if not similar_docs:
            try:
                table_ref = self.store.client._get_table_ref("documents", self.config.realm)
                rows = await self.store.client._fetch(f"SELECT realm, id, fqid, payload, created_at, updated_at FROM {table_ref} LIMIT $1", top_k)
                for r in rows:
                    v = Vertex(realm=r['realm'], id=str(r['id']), fqid=r['fqid'], payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']), created_at=r['created_at'], updated_at=r['updated_at'], table_name="documents", _client=self.store.client)
                    similar_docs.append((v, 0.0))
            except Exception:
                pass

        # 2. Gather graph relationship context from retrieved entities
        graph_triples = []
        for entity_vertex, dist in similar_entities:
            neighbors = await self.store.get_neighbors(entity_vertex.id)
            for edge, target in neighbors:
                subj_name = entity_vertex.payload.get("name", entity_vertex.id)
                obj_name = target.payload.get("name", target.id)
                rel_type = edge.relation_type
                graph_triples.append(f"({subj_name}) --[{rel_type}]--> ({obj_name})")

        # 3. Format document passage strings with metadata
        doc_passages = []
        retrieved_docs_output = []
        for v, d in similar_docs:
            meta = DocumentMetadata.from_dict(v.payload)
            meta_dict = meta.to_dict()
            meta_parts = []
            if meta.document:
                meta_parts.append(f"Document: {meta.document}")
            if meta.source:
                meta_parts.append(f"Source: {meta.source}")
            if meta.category:
                meta_parts.append(f"Category: {meta.category}")
            if meta.collection:
                meta_parts.append(f"Collection: {meta.collection}")
            if meta.page is not None:
                meta_parts.append(f"Page: {meta.page}")
            if meta.paragraph is not None:
                meta_parts.append(f"Paragraph: {meta.paragraph}")
            
            header = f" [{', '.join(meta_parts)}]" if meta_parts else ""
            doc_passages.append(f"- Chunk {v.id}{header}: {v.payload.get('text', '')}")
            retrieved_docs_output.append({
                "id": v.id,
                "text": v.payload.get("text"),
                "metadata": meta_dict
            })

        doc_context = "\n".join(doc_passages)
        entity_context = "\n".join([f"- Entity {v.payload.get('name')}: {v.payload.get('description')}" for v, d in similar_entities])
        graph_context = "\n".join([f"- {t}" for t in graph_triples])

        prompt = f"""You are a Knowledge Graph RAG assistant.
Use the following retrieved document passages and Knowledge Graph triples to answer the user question.

Retrieved Document Passages:
{doc_context or 'None'}

Retrieved Key Entities:
{entity_context or 'None'}

Retrieved Graph Relationships (Triples):
{graph_context or 'None'}

User Question: {question}

Synthesize a comprehensive, factual answer using both the document context and the graph relationships.
"""

        messages = [
            {"role": "system", "content": "You are a helpful Knowledge Graph RAG assistant."},
            {"role": "user", "content": prompt}
        ]

        answer = await self.llm.chat_completion(messages)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": retrieved_docs_output,
            "retrieved_entities": [v.payload.get("name") for v, d in similar_entities],
            "retrieved_graph_triples": graph_triples
        }
