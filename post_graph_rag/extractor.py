"""Domain-agnostic Knowledge Graph entity and triple extraction module."""
import json
import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from post_graph_rag.llm import LLMService

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    name: str = Field(..., description="Canonical entity name (e.g., 'PostgreSQL', 'DeepSeek-V3.2')")
    type: str = Field(..., description="Entity type/category (e.g., 'Software', 'Person', 'Organization', 'Concept', 'Location')")
    description: str = Field(..., description="Brief contextual summary of the entity")

class Triple(BaseModel):
    subject: str = Field(..., description="Subject entity name")
    predicate: str = Field(..., description="Normalized active relation predicate (e.g., 'uses', 'is_a', 'developed_by', 'part_of')")
    object: str = Field(..., description="Object entity name")
    description: Optional[str] = Field(None, description="Contextual note on the relationship")

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)

SYSTEM_PROMPT = """You are an expert, domain-agnostic Knowledge Graph Extractor.

Your task is to analyze text from ANY domain (technology, science, business, history, literature, medicine, law, etc.) and extract:
1. ENTITIES: Distinct, meaningful named entities or key concepts.
2. TRIPLES: Factual (Subject, Predicate, Object) relations connecting the extracted entities.

GUIDELINES FOR ENTITIES:
- Name: Clean, canonical entity name.
- Type: Broad category/type (e.g., 'Software', 'Person', 'Organization', 'Concept', 'Location', 'Event').
- Description: Brief summary of the entity's role in the text.

GUIDELINES FOR TRIPLES:
- Subject: Canonical name of the source entity (should match an extracted Entity name).
- Predicate: Clear, normalized relationship predicate in lowercase (e.g., 'uses', 'is_a', 'developed_by', 'located_in', 'causes', 'part_of', 'created_by').
- Object: Canonical name of the target entity (should match an extracted Entity name).
- Description: Additional contextual detail regarding the relation.

OUTPUT REQUIREMENTS:
Return your response formatted strictly according to the required schema.
"""

class GraphExtractor:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def extract_from_text(self, text: str) -> ExtractionResult:
        """Extract entities and triples from text content using LLM with generic fallback."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document Text:\n\n{text}"}
        ]

        result = await self.llm_service.chat_completion(messages, response_format=ExtractionResult)
        if isinstance(result, ExtractionResult) and (result.entities or result.triples):
            return result

        # Fallback parsing if JSON string was returned by LLM
        if isinstance(result, str) and result.strip():
            try:
                data = json.loads(result)
                return ExtractionResult(**data)
            except Exception:
                pass

        # Fully generic rule-based heuristic extraction fallback (no domain-specific logic)
        logger.info("Executing generic heuristic extraction fallback...")
        entities = []
        triples = []

        # Find proper nouns / capitalized terms
        words = re.findall(r'\b[A-Z][a-zA-Z0-9_\-]+\b', text)
        stop_words = {
            "The", "A", "An", "In", "On", "At", "By", "With", "From", "To", "And", "Or", "But",
            "This", "That", "These", "Those", "He", "She", "It", "They", "His", "Her", "Its",
            "Their", "Who", "What", "Where", "When", "Why", "How", "If", "Is", "Are", "Was", "Were"
        }
        
        seen_entities = {}
        for word in words:
            if word not in stop_words and len(word) > 1 and word not in seen_entities:
                e = Entity(name=word, type="Concept", description=f"Entity referenced in text: '{word}'")
                seen_entities[word] = e
                entities.append(e)

        # Extract basic co-occurring entity pairs as generic relations
        entity_names = list(seen_entities.keys())
        for i in range(len(entity_names) - 1):
            subj = entity_names[i]
            obj = entity_names[i + 1]
            triples.append(Triple(
                subject=subj,
                predicate="relates_to",
                object=obj,
                description=f"Generic relationship between {subj} and {obj}"
            ))

        return ExtractionResult(entities=entities, triples=triples)
