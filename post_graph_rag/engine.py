import inspect
import logging
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from post_graph import Vertex
from post_graph_rag.chunking import make_paragraph_chunker
from post_graph_rag.communities import default_detector, group_by_community
from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import RAGError
from post_graph_rag.reporting import CommunityReporter, report_to_text
from post_graph_rag.models import DocumentContext, DocumentMetadata, QueryParam
from post_graph_rag.llm import LLMService
from post_graph_rag.extractor import GraphExtractor, ExtractionResult
from post_graph_rag.graph_store import RAGGraphStore

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("mix", "local", "global", "hybrid", "naive", "bypass")

ENTITY_MODES = ("mix", "local", "hybrid")
DOCUMENT_MODES = ("mix", "local", "hybrid", "naive")
GLOBAL_MODES = ("global", "hybrid")


def _accepts_resolution(detector: Callable) -> bool:
    """Whether a community detector takes a ``resolution`` argument.

    Custom detectors are free to omit it; only resolution-based algorithms such
    as Leiden use one.
    """
    try:
        return "resolution" in inspect.signature(detector).parameters
    except (TypeError, ValueError):
        return False


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
    """Indexing and retrieval orchestrator.

    The LLM service, extractor and chunker can all be replaced, so callers can
    tune extraction or supply their own splitter without forking the library.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[LLMService] = None,
        extractor: Optional[GraphExtractor] = None,
        chunker: Optional[Callable[[str], List[str]]] = None,
        community_detector: Optional[Callable[..., Dict[str, int]]] = None,
        reporter: Optional[CommunityReporter] = None,
    ):
        self.config = config or RAGConfig()
        self.llm = llm or LLMService(self.config)
        self.extractor = extractor or GraphExtractor(
            self.llm,
            system_prompt=self.config.extraction_prompt,
            entity_types=self.config.entity_types or None,
            predicate_vocabulary=self.config.predicate_vocabulary or None,
            predicate_aliases=self.config.predicate_aliases or None,
            gleaning_passes=self.config.gleaning_passes,
            min_confidence=self.config.min_relation_confidence,
            drop_negated=self.config.drop_negated_relations,
            reject_possessive_entities=self.config.reject_possessive_entities,
        )
        self.chunker = chunker or make_paragraph_chunker(
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.chunk_overlap_chars,
        )
        self.community_detector = community_detector or default_detector
        self.reporter = reporter or CommunityReporter(
            self.llm, system_prompt=self.config.community_report_prompt
        )
        self.store = RAGGraphStore(self.config)

    async def initialize(self):
        """Initialize database connection and schema."""
        await self.store.connect()
        await self.store.initialize_schema()

    async def close(self):
        """Close database connection."""
        await self.store.close()

    # ---------------------------------------------------------------- indexing

    async def index_text(
        self,
        text: str,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk a whole document and index each chunk with running context.

        Each chunk is extracted with the document title and the canonical entity
        names found so far, so references that only resolve against earlier text
        ("he", "the engine") attach to the right entity instead of becoming
        vertices of their own.
        """
        meta_obj = metadata if isinstance(metadata, DocumentMetadata) else DocumentMetadata.from_dict(metadata or {})
        chunks = self.chunker(text)
        context = DocumentContext(title=meta_obj.document, source=meta_obj.source)

        results = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_meta = DocumentMetadata.from_dict({**meta_obj.to_dict(), "paragraph": idx})
            res = await self.index_document(chunk, metadata=chunk_meta, space=space, context=context)
            results.append(res)
            # Feed this chunk's canonical names forward as context.
            for name in res["entities"]:
                if name not in context.known_entities:
                    context.known_entities.append(name)
            del context.known_entities[: max(0, len(context.known_entities) - self.config.context_entity_limit)]
        return results

    async def index_document(
        self,
        text: str,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        space: Optional[str] = None,
        context: Optional[DocumentContext] = None,
    ) -> Dict[str, Any]:
        """Index one chunk: extract entities/triples, embed, and populate the graph."""
        meta_obj = metadata if isinstance(metadata, DocumentMetadata) else DocumentMetadata.from_dict(metadata or {})
        target_space = space or meta_obj.space or self.config.space

        if context is None and (meta_obj.document or meta_obj.source):
            context = DocumentContext(title=meta_obj.document, source=meta_obj.source)

        doc_emb = await self.llm.get_embedding(text)
        doc_vertex = await self.store.add_document(text, doc_emb, meta_obj, space=target_space)

        extraction: ExtractionResult = await self.extractor.extract_from_text(text, context=context)

        # One batched request instead of one round trip per entity.
        entity_texts = [
            f"{e.name} ({e.type}): {e.description}" + (f" Also known as: {', '.join(e.aliases)}." if e.aliases else "")
            for e in extraction.entities
        ]
        entity_embs = await self.llm.get_embeddings(entity_texts)

        entity_vertex_map: Dict[str, Vertex] = {}
        for entity, emb in zip(extraction.entities, entity_embs):
            e_vertex = await self.store.upsert_entity(
                name=entity.name,
                entity_type=entity.type,
                description=entity.description,
                embedding=emb,
                space=target_space,
                aliases=entity.aliases,
            )
            entity_vertex_map[entity.name.lower()] = e_vertex
            for alias in entity.aliases:
                entity_vertex_map.setdefault(alias.lower(), e_vertex)

        # Triple endpoints the extractor did not return as full entities.
        missing = []
        for triple in extraction.triples:
            for endpoint in (triple.subject, triple.object):
                if endpoint.lower() not in entity_vertex_map and endpoint not in missing:
                    missing.append(endpoint)
        if missing:
            stub_embs = await self.llm.get_embeddings(missing)
            for name, emb in zip(missing, stub_embs):
                entity_vertex_map[name.lower()] = await self.store.upsert_entity(
                    name, "Concept", "", emb, space=target_space
                )

        rel_embs: List[Optional[List[float]]] = [None] * len(extraction.triples)
        if self.config.embed_relations and extraction.triples:
            rel_texts = [
                f"{t.subject} {t.predicate} {t.object}. {t.description or ''}".strip()
                for t in extraction.triples
            ]
            rel_embs = list(await self.llm.get_embeddings(rel_texts))

        added_relations = []
        for triple, rel_emb in zip(extraction.triples, rel_embs):
            subj_vertex = entity_vertex_map[triple.subject.lower()]
            obj_vertex = entity_vertex_map[triple.object.lower()]
            edge = await self.store.add_relation(
                subj_vertex, obj_vertex, triple.predicate, triple.description,
                space=target_space, embedding=rel_emb,
                negated=triple.negated, confidence=triple.confidence,
            )
            added_relations.append(edge)

        # Link the chunk to every entity it mentions, populating doc_mentions.
        mentioned = {v.id: v for v in entity_vertex_map.values()}
        mentions = 0
        for e_vertex in mentioned.values():
            if await self.store.add_doc_mention(doc_vertex, e_vertex, space=target_space):
                mentions += 1

        return {
            "document_id": doc_vertex.id,
            "entities_extracted": len(extraction.entities),
            "triples_extracted": len(extraction.triples),
            "relations_added": len(added_relations),
            "mentions_added": mentions,
            "negated_relations": sum(1 for t in extraction.triples if t.negated),
            "entities": [e.name for e in extraction.entities],
            "metadata": meta_obj.to_dict()
        }

    # ------------------------------------------------------------- communities

    async def build_communities(self, space: Optional[str] = None) -> Dict[str, Any]:
        """Cluster the entity graph and summarise each cluster.

        Run this after indexing. Corpus-level questions are answered from these
        summaries: no single passage contains the answer to "what are the main
        themes here?", so retrieving passages cannot produce one.

        Communities are derived data and are rebuilt wholesale, replacing any
        previous clustering for the space.
        """
        target_space = space or self.config.space
        entities, relations = await self.store.graph_snapshot(space=target_space)
        if not entities:
            return {"communities": 0, "entities": 0, "skipped": 0, "reason": "no entities"}

        by_id = {e["id"]: e for e in entities}
        edges = [
            (
                r["from_id"], r["to_id"],
                float(r.get("weight", 1)) * (self.config.negated_relation_weight if r["negated"] else 1.0),
            )
            for r in relations
            if r["from_id"] in by_id and r["to_id"] in by_id
        ]

        assignment = self.community_detector(
            [e["id"] for e in entities], edges, resolution=self.config.community_resolution
        ) if _accepts_resolution(self.community_detector) else self.community_detector(
            [e["id"] for e in entities], edges
        )

        groups = group_by_community(assignment, min_size=self.config.community_min_size)
        # Largest first, so a truncated build still covers the most significant clusters.
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        capped = ordered[: self.config.max_communities]
        if len(ordered) > len(capped):
            logger.warning(
                "Detected %d communities; summarising the %d largest (max_communities).",
                len(ordered), len(capped),
            )

        await self.store.clear_communities(space=target_space)

        rels_by_member: Dict[str, List[Dict[str, Any]]] = {}
        for r in relations:
            rels_by_member.setdefault(r["from_id"], []).append(r)

        built, skipped = 0, 0
        used_titles: Dict[str, int] = {}
        for community_id, member_ids in capped:
            members = set(member_ids)
            member_entities = [by_id[m] for m in member_ids if m in by_id]
            member_relations = [
                {
                    "src": by_id[r["from_id"]]["name"], "tgt": by_id[r["to_id"]]["name"],
                    "predicate": r["predicate"], "description": r["description"],
                    "weight": r["weight"], "negated": r["negated"],
                }
                for m in member_ids
                for r in rels_by_member.get(m, [])
                if r["to_id"] in members
            ]

            try:
                report = await self.reporter.summarise(member_entities, member_relations)
                text = report_to_text(report)
                embedding = await self.llm.get_embedding(text)
                await self.store.add_community(
                    key=f"c{community_id}",
                    title=self._disambiguate_title(report.title, member_entities, used_titles),
                    summary=report.summary,
                    embedding=embedding,
                    rating=report.rating,
                    findings=[f.model_dump() for f in report.findings],
                    entity_ids=member_ids,
                    space=target_space,
                )
                built += 1
            except RAGError as e:
                # One unusable report should not abandon the whole build.
                logger.warning("Skipping community %s: %s", community_id, e)
                skipped += 1

        return {
            "communities": built,
            "skipped": skipped,
            "detected": len(ordered),
            "entities": len(entities),
            "relations": len(relations),
        }

    @staticmethod
    def _disambiguate_title(
        title: str, members: List[Dict[str, Any]], used: Dict[str, int]
    ) -> str:
        """Make community titles unique within a build.

        Reports are generated independently, so two distinct clusters can be
        handed the same title. Retrieval then shows the same label twice with no
        way to tell which subgraph is which.
        """
        base = (title or "Community").strip()
        seen = used.get(base, 0)
        used[base] = seen + 1
        if seen == 0:
            return base

        # Qualify with a member that is not already named in the title.
        for entity in members:
            name = str(entity.get("name") or "")
            if name and name.lower() not in base.lower():
                return f"{base} ({name})"
        return f"{base} #{seen + 1}"

    # --------------------------------------------------------------- retrieval

    async def query_data(self, question: str, param: Optional[QueryParam] = None) -> Dict[str, Any]:
        """Structured data retrieval API: returns raw retrieved entities, relationships, chunks, and metadata without LLM synthesis."""
        p = param or QueryParam()
        mode = p.mode.lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{p.mode}'. Expected one of {RETRIEVAL_MODES}.")
        target_space = p.space or self.config.space

        if mode == "bypass":
            return {
                "status": "success",
                "message": "Direct query (bypass mode)",
                "data": {"entities": [], "relationships": [], "chunks": [], "references": [], "communities": []},
                "metadata": {"query_mode": mode, "keywords": {"high_level": [], "low_level": []}}
            }

        if not p.hl_keywords and not p.ll_keywords:
            kw_res = await self.extractor.extract_keywords(question)
            p.hl_keywords = kw_res.high_level_keywords
            p.ll_keywords = kw_res.low_level_keywords

        query_vec = await self.llm.get_embedding(question)

        # Low-level keywords name concrete entities, so they sharpen entity search.
        # High-level keywords describe themes, so they steer relation ranking.
        entity_vec = query_vec
        if p.ll_keywords:
            entity_vec = await self.llm.get_embedding(
                f"{question}\nEntities: {', '.join(p.ll_keywords)}"
            )

        similar_entities: List[Tuple[Vertex, float]] = []
        similar_docs: List[Tuple[Vertex, float]] = []
        graph_triples: List[Dict[str, Any]] = []

        if mode in ENTITY_MODES:
            similar_entities = await self.store.search_similar_entities(
                entity_vec, top_k=p.top_k, space=target_space
            )
        if mode in DOCUMENT_MODES:
            similar_docs = await self.store.search_similar_documents(
                query_vec, top_k=p.top_k, space=target_space
            )

        if mode in ENTITY_MODES:
            for entity_vertex, _dist in similar_entities:
                for edge, target in await self.store.get_neighbors(entity_vertex.id, space=target_space):
                    graph_triples.append(self._format_triple(edge, entity_vertex, target))

            # Pull in passages that mention the matched entities. A question can
            # match an entity by name while the passage explaining it uses none
            # of the query's wording, including passages in other documents.
            if self.config.expand_chunks_via_mentions and similar_entities and mode in DOCUMENT_MODES:
                have = [v.id for v, _ in similar_docs]
                extra = await self.store.find_chunks_mentioning(
                    [v.id for v, _ in similar_entities],
                    limit=p.top_k,
                    space=target_space,
                    exclude_ids=have,
                )
                similar_docs.extend((v, 1.0) for v in extra)

        communities: List[Dict[str, Any]] = []
        if mode in GLOBAL_MODES:
            communities = await self._retrieve_communities(query_vec, p, target_space)
            # Relations remain useful alongside reports, but when communities
            # exist they carry the corpus-level answer and relations are support.
            graph_triples.extend(await self._global_relations(p, query_vec, target_space))

        graph_triples = self._dedupe_triples(graph_triples)

        formatted_entities = [{
            "entity_name": v.payload.get("name"),
            "entity_type": v.payload.get("type"),
            "description": v.payload.get("description")
        } for v, _dist in similar_entities]

        formatted_chunks = [{
            "chunk_id": v.id,
            "content": v.payload.get("text"),
            "metadata": DocumentMetadata.from_dict(v.payload).to_dict()
        } for v, _dist in similar_docs]

        references = [{
            "reference_id": f"[{idx + 1}]",
            "document": v.payload.get("document") or f"Doc Chunk {v.id}"
        } for idx, (v, _dist) in enumerate(similar_docs)]

        return {
            "status": "success",
            "message": "Query data retrieved successfully",
            "data": {
                "entities": formatted_entities,
                "relationships": graph_triples,
                "chunks": formatted_chunks,
                "references": references,
                "communities": communities
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
                    "final_chunks_count": len(similar_docs),
                    "communities_found": len(communities)
                }
            }
        }

    async def _retrieve_communities(
        self, query_vec: List[float], p: QueryParam, space: str
    ) -> List[Dict[str, Any]]:
        """Retrieve community reports relevant to the question.

        Returns an empty list when no communities have been built, so global mode
        degrades to relation ranking rather than failing.
        """
        try:
            hits = await self.store.search_similar_communities(query_vec, top_k=p.top_k, space=space)
        except Exception as e:
            logger.warning("Community search failed (%s); continuing without reports.", e)
            return []

        out = []
        for vertex, distance in hits:
            payload = vertex.payload or {}
            out.append({
                "community_id": vertex.id,
                "title": payload.get("title"),
                "summary": payload.get("summary"),
                "findings": payload.get("findings") or [],
                "rating": payload.get("rating", 5.0),
                "size": payload.get("size", 0),
                "distance": distance,
            })
        # Most relevant first, then most important.
        out.sort(key=lambda c: (c["distance"], -float(c.get("rating") or 0)))
        return out

    async def _global_relations(self, p: QueryParam, query_vec: List[float], space: str) -> List[Dict[str, Any]]:
        """Collect graph-wide relations for 'global'/'hybrid' modes.

        Uses relation embeddings when ``embed_relations`` is enabled; otherwise
        enumerates relations and ranks them by keyword overlap, which is still
        far better than returning whichever rows were read first.
        """
        limit = p.top_k * 5

        if self.config.embed_relations:
            hits = await self.store.search_similar_relations(query_vec, top_k=limit, space=space)
            if hits:
                resolved = []
                for edge, _dist in hits:
                    from_v, to_v = await self.store.get_relation_endpoints(edge)
                    if from_v is None or to_v is None:
                        continue
                    resolved.append(self._format_triple(edge, from_v, to_v))
                return resolved

        candidates = [
            self._format_triple(edge, from_v, to_v)
            for edge, from_v, to_v in await self.store.get_all_relations(limit=limit, space=space)
        ]
        return self._rank_by_keywords(candidates, p.hl_keywords + p.ll_keywords, limit)

    @staticmethod
    def _format_triple(edge: Any, from_v: Vertex, to_v: Vertex) -> Dict[str, Any]:
        payload = edge.payload or {}
        return {
            "src_id": from_v.payload.get("name", from_v.id),
            "tgt_id": to_v.payload.get("name", to_v.id),
            "relation_type": edge.relation_type,
            "description": payload.get("description", ""),
            "weight": payload.get("weight", 1),
            "negated": bool(payload.get("negated", False)),
            "confidence": float(payload.get("confidence", 1.0)),
        }

    @staticmethod
    def _rank_by_keywords(triples: List[Dict[str, Any]], keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        """Order global-mode relations by keyword overlap.

        Global mode has no per-relation vector to search against, so without this
        it returns whatever ``get_all_relations`` happened to read first.
        """
        terms = {k.lower() for k in keywords if k}

        def score(t: Dict[str, Any]) -> tuple:
            haystack = " ".join([
                str(t.get("src_id", "")), str(t.get("tgt_id", "")),
                str(t.get("relation_type", "")), str(t.get("description", ""))
            ]).lower()
            overlap = sum(1 for term in terms if term in haystack)
            # Weight breaks ties: a relation corroborated by several chunks is
            # more likely to matter than one seen once.
            return (overlap, int(t.get("weight", 1)))

        return sorted(triples, key=score, reverse=True)[:limit]

    @staticmethod
    def _dedupe_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relations that entity-hop and global passes both found."""
        seen = set()
        out = []
        for t in triples:
            key = (str(t.get("src_id")).lower(), str(t.get("relation_type")).lower(), str(t.get("tgt_id")).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    # --------------------------------------------------------------- synthesis

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
            return {
                "question": question, "answer": answer, "mode": mode,
                "retrieved_documents": [], "retrieved_entities": [], "retrieved_graph_triples": []
            }

        data = data_res["data"]
        doc_passages = [
            f"Chunk [{idx + 1}] ({chunk['metadata'].get('document', 'Doc')}): {chunk['content']}"
            for idx, chunk in enumerate(data["chunks"])
        ]
        entity_passages = [
            f"- Entity {e['entity_name']} ({e['entity_type']}): {e['description']}"
            for e in data["entities"]
        ]
        # Negated relations are rendered explicitly. Stored as a positive
        # predicate with a flag, they would otherwise read to the model as an
        # assertion that the relation holds.
        triple_passages = [
            f"- ({r['src_id']}) --[{'NOT ' if r.get('negated') else ''}{r['relation_type']}"
            f" (weight={r['weight']})]--> ({r['tgt_id']}): {r['description']}"
            for r in data["relationships"]
        ]

        # Community reports carry corpus-level structure that no single passage
        # holds, so they lead the context when present.
        community_passages = []
        for c in data.get("communities", []):
            findings = " ".join(
                f"{f.get('summary', '')}: {f.get('explanation', '')}" for f in (c.get("findings") or [])
            )
            community_passages.append(
                f"- {c.get('title')} (importance {c.get('rating')}, {c.get('size')} entities): "
                f"{c.get('summary')} {findings}".strip()
            )
        community_context = (
            _truncate_by_tokens(community_passages, p.max_total_tokens // 3)
            if community_passages else "None"
        )

        doc_context = _truncate_by_tokens(doc_passages, p.max_total_tokens // 2) if doc_passages else "None"
        entity_context = _truncate_by_tokens(entity_passages, p.max_entity_tokens) if entity_passages else "None"
        graph_context = _truncate_by_tokens(triple_passages, p.max_relation_tokens) if triple_passages else "None"

        ref_list_str = "\n".join([
            f"- [{idx + 1}] {chunk['metadata'].get('document', 'Document Chunk ' + str(chunk['chunk_id']))}"
            for idx, chunk in enumerate(data["chunks"])
        ])

        prompt = f"""---Role---
You are an expert AI assistant synthesizing information from a Knowledge Base operating in '{mode}' retrieval mode.
Your answer should be formatted in: {p.response_type}.

---Retrieved Context---

Knowledge Base Themes (summaries of clustered subgraphs):
{community_context}

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
3. If the retrieved context does not answer the question, say so plainly instead of guessing.
4. Format your response cleanly using Markdown.
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
            "retrieved_graph_triples": [
                f"({r['src_id']}) --[{r['relation_type']}]--> ({r['tgt_id']})"
                for r in data["relationships"]
            ],
            "retrieved_communities": [c["title"] for c in data.get("communities", [])],
            "references": data["references"]
        }
