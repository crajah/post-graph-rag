import json
import logging
from typing import List, Dict, Any, Optional, Union
from post_graph import Vertex
from post_graph_rag.config import RAGConfig
from post_graph_rag.models import DocumentMetadata, QueryParam, KeywordResult
from post_graph_rag.llm import LLMService
from post_graph_rag.extractor import GraphExtractor, ExtractionResult
from post_graph_rag.graph_store import RAGGraphStore

logger = logging.getLogger(__name__)

def _truncate_by_tokens(text_items: List[str], max_tokens: int) -> str:
    """Helper to truncate a list of string passages to respect max token budget."""
    current_tokens = 0
    selected = []
    for item in text_items:
        item_tokens = max(1, len(item) // 4)
        if current_tokens + item_tokens > max_tokens:
            break
        selected.append(item)
        current_tokens += item_tokens
    return "\n".join(selected)

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

    async def query_data(self, question: str, param: Optional[QueryParam] = None) -> Dict[str, Any]:
        """Structured data retrieval API: returns raw retrieved entities, relationships, chunks, and metadata without LLM synthesis."""
        p = param or QueryParam()
        mode = p.mode.lower()

        # Dual-level keyword extraction
        if not p.hl_keywords and not p.ll_keywords and mode != "bypass":
            kw_res = await self.extractor.extract_keywords(question)
            p.hl_keywords = kw_res.high_level_keywords
            p.ll_keywords = kw_res.low_level_keywords

        if mode == "bypass":
            return {
                "status": "success",
                "message": "Direct query (bypass mode)",
                "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
                "metadata": {"query_mode": mode, "keywords": {"high_level": [], "low_level": []}}
            }

        query_vec = await self.llm.get_embedding(question)
        similar_entities = []
        similar_docs = []
        global_relations = []

        if mode in ("mix", "local", "hybrid"):
            similar_entities = await self.store.search_similar_entities(query_vec, top_k=p.top_k)
        if mode in ("mix", "local", "hybrid", "naive"):
            similar_docs = await self.store.search_similar_documents(query_vec, top_k=p.top_k)
        if mode in ("global", "hybrid"):
            global_relations = await self.store.get_all_relations(limit=p.top_k * 5)

        # Fallback if empty vector hits
        if mode in ("mix", "local", "hybrid") and not similar_entities:
            try:
                table_ref = self.store.client._get_table_ref("entities", self.config.realm)
                rows = await self.store.client._fetch(f"SELECT realm, id, fqid, payload, created_at, updated_at FROM {table_ref} LIMIT $1", p.top_k * 3)
                for r in rows:
                    v = Vertex(realm=r['realm'], id=str(r['id']), fqid=r['fqid'], payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']), created_at=r['created_at'], updated_at=r['updated_at'], table_name="entities", _client=self.store.client)
                    similar_entities.append((v, 0.0))
            except Exception:
                pass

        if mode in ("mix", "local", "hybrid", "naive") and not similar_docs:
            try:
                table_ref = self.store.client._get_table_ref("documents", self.config.realm)
                rows = await self.store.client._fetch(f"SELECT realm, id, fqid, payload, created_at, updated_at FROM {table_ref} LIMIT $1", p.top_k)
                for r in rows:
                    v = Vertex(realm=r['realm'], id=str(r['id']), fqid=r['fqid'], payload=r['payload'] if isinstance(r['payload'], dict) else json.loads(r['payload']), created_at=r['created_at'], updated_at=r['updated_at'], table_name="documents", _client=self.store.client)
                    similar_docs.append((v, 0.0))
            except Exception:
                pass

        # Gather graph relationships from entities
        graph_triples = []
        if mode in ("mix", "local", "hybrid"):
            for entity_vertex, dist in similar_entities:
                neighbors = await self.store.get_neighbors(entity_vertex.id)
                for edge, target in neighbors:
                    subj_name = entity_vertex.payload.get("name", entity_vertex.id)
                    obj_name = target.payload.get("name", target.id)
                    rel_type = edge.relation_type
                    graph_triples.append({
                        "src_id": subj_name,
                        "tgt_id": obj_name,
                        "relation_type": rel_type,
                        "description": edge.payload.get("description", ""),
                        "weight": edge.payload.get("weight", 1)
                    })

        if mode in ("global", "hybrid"):
            for edge, from_v, to_v in global_relations:
                subj_name = from_v.payload.get("name", from_v.id)
                obj_name = to_v.payload.get("name", to_v.id)
                graph_triples.append({
                    "src_id": subj_name,
                    "tgt_id": obj_name,
                    "relation_type": edge.relation_type,
                    "description": edge.payload.get("description", ""),
                    "weight": edge.payload.get("weight", 1)
                })

        formatted_entities = [{
            "entity_name": v.payload.get("name"),
            "entity_type": v.payload.get("type"),
            "description": v.payload.get("description")
        } for v, dist in similar_entities]

        formatted_chunks = [{
            "chunk_id": v.id,
            "content": v.payload.get("text"),
            "metadata": DocumentMetadata.from_dict(v.payload).to_dict()
        } for v, dist in similar_docs]

        references = [{
            "reference_id": f"[{idx + 1}]",
            "document": v.payload.get("document") or f"Doc Chunk {v.id}"
        } for idx, (v, dist) in enumerate(similar_docs)]

        return {
            "status": "success",
            "message": "Query data retrieved successfully",
            "data": {
                "entities": formatted_entities,
                "relationships": graph_triples,
                "chunks": formatted_chunks,
                "references": references
            },
            "metadata": {
                "query_mode": mode,
                "keywords": {
                    "high_level": p.hl_keywords,
                    "low_level": p.ll_keywords
                },
                "processing_info": {
                    "total_entities_found": len(similar_entities),
                    "total_relations_found": len(graph_triples),
                    "final_chunks_count": len(similar_docs)
                }
            }
        }

    async def query(
        self,
        question: str,
        param: Optional[QueryParam] = None,
        top_k: Optional[int] = None
    ) -> Union[Dict[str, Any], Any]:
        """Execute RAG query with support for multiple retrieval modes, dual-level keywords, token budgeting, and streaming."""
        p = param or QueryParam()
        if top_k is not None:
            p.top_k = top_k

        if p.only_need_context:
            return await self.query_data(question, param=p)

        data_res = await self.query_data(question, param=p)
        mode = p.mode.lower()

        if mode == "bypass":
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            if p.conversation_history:
                messages.extend(p.conversation_history)
            messages.append({"role": "user", "content": question})

            if p.stream:
                return self.llm.chat_completion_stream(messages)
            answer = await self.llm.chat_completion(messages)
            return {"question": question, "answer": answer, "mode": mode, "retrieved_documents": [], "retrieved_entities": [], "retrieved_graph_triples": []}

        data = data_res["data"]
        doc_passages = []
        for idx, chunk in enumerate(data["chunks"]):
            ref_id = f"[{idx + 1}]"
            doc_passages.append(f"Chunk {ref_id} ({chunk['metadata'].get('document', 'Doc')}): {chunk['content']}")

        entity_passages = [f"- Entity {e['entity_name']} ({e['entity_type']}): {e['description']}" for e in data["entities"]]
        triple_passages = [f"- ({r['src_id']}) --[{r['relation_type']} (weight={r['weight']})]--> ({r['tgt_id']}): {r['description']}" for r in data["relationships"]]

        doc_context = _truncate_by_tokens(doc_passages, p.max_total_tokens // 2) if doc_passages else "None"
        entity_context = _truncate_by_tokens(entity_passages, p.max_entity_tokens) if entity_passages else "None"
        graph_context = _truncate_by_tokens(triple_passages, p.max_relation_tokens) if triple_passages else "None"

        ref_list_str = "\n".join([f"- [{idx + 1}] {chunk['metadata'].get('document', 'Document Chunk ' + str(chunk['chunk_id']))}" for idx, chunk in enumerate(data["chunks"])])

        prompt = f"""---Role---
You are an expert AI assistant synthesizing information from a Knowledge Base operating in '{mode}' retrieval mode.
Your answer should be formatted in: {p.response_type}.

---Retrieved Context---

Retrieved Document Passages:
{doc_context}

Retrieved Key Entities:
{entity_context}

Retrieved Graph Relationships:
{graph_context}

Reference Document List:
{ref_list_str or 'None'}

---User Question---
{question}

---Instructions---
1. Answer the question directly using facts from the context.
2. If citing specific document chunks, refer to them using reference IDs like [1], [2].
3. Format your response cleanly using Markdown.
"""

        messages = [{"role": "system", "content": "You are a helpful Knowledge Graph RAG assistant."}]
        if p.conversation_history:
            messages.extend(p.conversation_history)
        messages.append({"role": "user", "content": prompt})

        if p.stream:
            return self.llm.chat_completion_stream(messages)

        answer = await self.llm.chat_completion(messages)

        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            "keywords": data_res["metadata"]["keywords"],
            "retrieved_documents": data["chunks"],
            "retrieved_entities": [e["entity_name"] for e in data["entities"]],
            "retrieved_graph_triples": [f"({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})" for r in data["relationships"]],
            "references": data["references"]
        }
