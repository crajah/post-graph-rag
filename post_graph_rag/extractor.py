"""Domain-agnostic Knowledge Graph entity and triple extraction module."""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from post_graph_rag.errors import ExtractionError
from post_graph_rag.llm import LLMService
from post_graph_rag.models import DocumentContext, KeywordResult

logger = logging.getLogger(__name__)

# Predicates that carry no information about *how* two entities are related.
# A graph edge labelled with one of these is indistinguishable from no edge at
# all, so they are rejected rather than stored.
VAGUE_PREDICATES = {
    "relates_to", "related_to", "relation", "relates", "associated_with",
    "connected_to", "linked_to", "is_related", "has_relation", "see_also",
    "mentioned_with", "appears_with",
}

# Leading auxiliaries carry tense, not meaning. Stripping them collapses
# 'was_appointed_knight_of' and 'appointed_knight_of' onto one predicate.
TENSE_PREFIXES = (
    "was_", "were_", "is_", "are_", "been_", "being_", "be_",
    "has_", "have_", "had_", "did_", "does_", "do_",
)

DEFAULT_ENTITY_TYPES = [
    "Person", "Organization", "Location", "Event", "Work", "Product",
    "Technology", "Concept",
]

# Names that are references rather than entities. They vary by chunk, so storing
# them produces vertices like 'his father' that can never resolve or connect.
PRONOMINAL_PREFIXES = (
    "his ", "her ", "their ", "its ", "our ", "your ", "my ",
    "this ", "that ", "these ", "those ", "the incident", "the above",
)
PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "theirs",
    "i", "we", "us", "you", "who", "whom", "someone", "anyone", "the author",
}

# "Ada Lovelace and Charles Babbage" is two entities, not one. Matches only a
# bare conjunction of proper-noun phrases, so real titles containing 'and'
# ("The Thrilling Adventures of Lovelace and Babbage") are left alone.
CONJUNCTION_ENTITY = re.compile(
    r"^[A-Z][\w.\-]*(?: [A-Z][\w.\-]*)* and [A-Z][\w.\-]*(?: [A-Z][\w.\-]*)*$"
)

# "Babbage's father", "Ada Lovelace's affair" — a possessive whose head is a
# common noun usually describes a role or attribute rather than a nameable
# entity. This is OFF by default: on real text it also catches legitimate named
# things such as "Ampère's force law" and "Menabrea's paper", so roughly half its
# hits are false positives. Callers who prefer a tighter graph can opt in.
POSSESSIVE_COMMON_NOUN = re.compile(r"^[A-Z][\w.\-]*(?: [A-Z][\w.\-]*)*'s [a-z]")


# Entity resolution depends on a unique btree index over the entity name, and
# Postgres refuses any index key beyond ~2704 bytes. Long before that limit a
# "name" has stopped being one: dense corpora such as SEC filings occasionally
# hand back an entire table row or sentence as an entity. Rejecting those keeps
# the graph clean and keeps a single bad extraction from aborting a whole run.
MAX_ENTITY_NAME_CHARS = 200

# A bare quantity is a measurement, not a thing that can hold relationships.
# Financial filings provoke this constantly: "$18.4 billion" and "$1,326" arrive
# as entity names, and because every filing reports different figures they
# fragment the graph — the same claim never recurs as the same pair of vertices,
# so nothing can supersede anything. Deliberately narrow: it matches only names
# that are *entirely* a figure, leaving "717 aircraft" and "$177 tax benefit"
# alone, since those name a real subject the text goes on to talk about.
# The abbreviated scale suffixes (k/m/bn) are only safe to strip when a currency
# symbol marks the string as money: bare "3M" is as likely to be the company.
BARE_QUANTITY = re.compile(
    r"""^[-+(]?\s*
        (?:
            [$€£¥]\s*\d[\d,.\s]*\s*(?:k|m|bn|million|billion|trillion)?
          | \d[\d,.\s]*\s*(?:%|percent|bps|million|billion|trillion)?\s*[$€£¥]?
        )
        \s*\)?$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_phrase_not_entity(name: str, reject_possessive: bool = False) -> bool:
    """True for compound phrases that name a relationship rather than an entity."""
    n = (name or "").strip()
    if not n:
        return False
    if len(n) > MAX_ENTITY_NAME_CHARS:
        return True
    if BARE_QUANTITY.match(n):
        return True
    if CONJUNCTION_ENTITY.match(n):
        return True
    return bool(reject_possessive and POSSESSIVE_COMMON_NOUN.match(n))


class Entity(BaseModel):
    name: str = Field(..., description="Canonical entity name, the fullest form available")
    type: str = Field(..., description="Entity type/category")
    description: str = Field(..., description="Brief contextual summary of the entity")
    aliases: List[str] = Field(
        default_factory=list,
        description="Other surface forms for this same entity in the text (surnames, short forms, titles, former names)",
    )


class Triple(BaseModel):
    subject: str = Field(..., description="Subject entity name")
    predicate: str = Field(..., description="Specific active relation predicate drawn from the text")
    object: str = Field(..., description="Object entity name")
    description: Optional[str] = Field(default=None, description="Contextual note on the relationship")
    negated: bool = Field(
        default=False,
        description="True when the text states this relation does NOT hold",
    )
    confidence: float = Field(
        default=1.0,
        description="0.0-1.0 confidence that the text actually asserts this relation",
    )
    valid_from: Optional[str] = Field(
        default=None,
        description=(
            "When the relation STARTED, only if the text states it. ISO-8601 or a bare "
            "year ('1625', '1625-06'). Leave null when the text gives no start."
        ),
    )
    valid_to: Optional[str] = Field(
        default=None,
        description=(
            "When the relation ENDED, only if the text states it. ISO-8601 or a bare "
            "year. Leave null when the relation is open-ended or no end is stated."
        ),
    )


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)


class KeywordResultSchema(BaseModel):
    high_level_keywords: List[str] = Field(default_factory=list, description="High-level overarching themes, concepts, or intent terms")
    low_level_keywords: List[str] = Field(default_factory=list, description="Specific entities, proper nouns, jargon, or concrete items")


class ContradictionResultSchema(BaseModel):
    contradicted_ids: List[str] = Field(
        default_factory=list,
        description="IDs of existing facts the new fact contradicts. Empty if none.")


CONTRADICTION_SYSTEM_PROMPT = """You decide which previously recorded facts a new fact makes false.

You are given one NEW FACT and a numbered list of EXISTING FACTS about the same subject.
Return the IDs of existing facts that the new fact CONTRADICTS.

A fact is contradicted only when both cannot be true of the same subject at the same time:
- "lives in Paris" vs "lives in Berlin" -> contradicted, a person lives in one place
- "prefers tea" vs "prefers coffee" -> contradicted, a stated preference replaces the old one
- "works at Acme" vs "works at Globex" -> contradicted unless the text supports holding both

A fact is NOT contradicted when it merely differs, adds detail, or concerns another aspect:
- "lives in Paris" vs "visited Berlin" -> not contradicted, different relations
- "owns a car" vs "owns a bicycle" -> not contradicted, both can hold
- "likes jazz" vs "likes jazz and blues" -> not contradicted, one refines the other
- Facts whose stated validity periods do not overlap -> NOT contradicted, both held, at
  different times. That is what the periods are for.

Be conservative. Marking a fact contradicted retracts it from every future answer, so
when the two can be read as coexisting, leave it alone. Returning an empty list is a
correct and common answer.

Respond with JSON: {"contradicted_ids": ["3", "7"]}
"""


BASE_SYSTEM_PROMPT = """You are an expert, domain-agnostic Knowledge Graph Extractor.

Your task is to analyze text from ANY domain (technology, science, business, history, literature, medicine, law, etc.) and extract:
1. ENTITIES: Distinct, meaningful named entities or key concepts.
2. TRIPLES: Factual (Subject, Predicate, Object) relations connecting the extracted entities.

GUIDELINES FOR ENTITIES:
- Name: the FULLEST canonical form of the entity. Prefer 'Charles Babbage' over 'Babbage',
  'Ada Lovelace' over 'Ada' or 'Countess of Lovelace'. If the chunk only gives a short form
  but the document context names the full form, use the full form.
- Aliases: every OTHER surface form used for that same entity — surnames, given names,
  titles, former names, abbreviations, possessive forms. Example: for 'Charles Babbage',
  aliases ['Babbage']. For 'Ada Lovelace', aliases ['Ada', 'Augusta Ada King', 'Countess of Lovelace'].
- NEVER emit a pronoun or relative reference as an entity ('he', 'his father', 'the company').
  Resolve it to the named entity it refers to using the document context, or omit it entirely.
- NEVER emit a conjunction of two entities as one entity ('Ada Lovelace and Charles Babbage').
  Emit them separately and connect them with a triple.
- NEVER emit a possessive role phrase as an entity ("Babbage's father", "Babbage's design").
  Name the referent if the text gives it, otherwise omit it.
- Emit the STABLE entity that could be named again in a DIFFERENT document about the
  same subject. Never bundle an entity together with a change, measurement, period,
  outcome or event into a single name.
    WRONG: 'Boeing Commercial Airplanes revenue increase', 'Q4 2005 net loss',
           '737 programme production impacts', 'Lovelace's 1843 translation work'
    RIGHT: 'Boeing Commercial Airplanes', 'net loss', '737 programme', 'Ada Lovelace'
  Put the movement, period and magnitude in the TRIPLE and its description instead:
  ('Boeing Commercial Airplanes')-[increases_revenue]->('Revenue'), described as
  "revenues rose 15 percent in 2006".
  TEST BEFORE EMITTING: would this exact name plausibly appear in another document
  on this subject? If it names a one-off event, figure or period-specific movement,
  it is not an entity — it is a relation.
- NEVER emit a bare figure, amount or date as an entity ('$18.4 billion', '2.5%',
  '$1,326'). A quantity is the VALUE of a relation, not a thing that holds relations;
  it belongs in the relation description.
- Type: {type_guidance}
- Description: Brief summary of the entity's role in THIS text.

GUIDELINES FOR TRIPLES:
- Subject/Object: canonical names that match extracted Entity names exactly.
- Predicate: the SPECIFIC relationship stated in the text, lowercase with underscores.
  Use the present-tense base form: 'appoint_knight_of', not 'was_appointed_knight_of'.
  FORBIDDEN: vague connectors such as 'relates_to', 'associated_with', 'connected_to', 'linked_to'.
  If the text does not state a specific relationship between two entities, emit NO triple for that pair.
  Never emit a triple merely because two entities appear near each other.
{predicate_guidance}
- negated: set TRUE when the text states the relation does NOT hold ("never met", "was not
  the son of", "had no contact with"). Do NOT invert the predicate into a negative phrase
  like 'did_not_have_relationship_with'; use the positive predicate and set negated=true.
- confidence: 1.0 when the text asserts the relation plainly, lower when it is hedged,
  speculative, or attributed to disputed sources.
- Description: the supporting detail from the text for this relation.
{validity_guidance}

OUTPUT REQUIREMENTS:
Return your response formatted strictly according to the required schema.
"""

GLEANING_PROMPT = """Some entities and relations were missed in the previous extraction.

ALREADY EXTRACTED ENTITIES: {entities}
ALREADY EXTRACTED RELATIONS: {triples}

Re-read the text and extract ONLY entities and relations that are MISSING from the lists above.
Apply the same rules. Return an empty result if nothing was missed. Do not repeat anything already listed.
"""

KEYWORD_SYSTEM_PROMPT = """You are an expert dual-level keyword extractor for a Retrieval-Augmented Generation (RAG) system.
Your job is to analyze the user query and extract:
1. high_level_keywords: Overarching themes, domain concepts, or core intent.
2. low_level_keywords: Specific entities, proper nouns, technical terms, or concrete items.

Return your response strictly adhering to the JSON schema.
"""


def normalise_predicate(predicate: str) -> str:
    """Lowercase, underscore-join, and strip leading tense auxiliaries."""
    p = re.sub(r"[\s\-]+", "_", (predicate or "").strip().lower())
    p = re.sub(r"_+", "_", p).strip("_")
    for prefix in TENSE_PREFIXES:
        if p.startswith(prefix) and len(p) > len(prefix) + 2:
            p = p[len(prefix):]
            break
    return p


def _clean_date(value: Optional[str]) -> Optional[str]:
    """Normalise a stated date to ISO-8601-ish text, or None.

    Accepts a bare year, year-month, or full date. Anything vaguer ("later",
    "in his youth") is discarded rather than coerced: a wrong date is worse than
    no date, because no date already means "always valid".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown", "n/a", "-"}:
        return None
    match = re.match(r"^(-?\d{1,4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$", text)
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    out = f"{int(year):04d}"
    if month:
        out += f"-{int(month):02d}"
        if day:
            out += f"-{int(day):02d}"
    return out


def date_sort_key(value: Optional[str]) -> str:
    """Pad a partial date so string comparison orders it correctly.

    '1625' must compare below '1625-06-01', so a bare year becomes '1625-01-01'.
    """
    if not value:
        return ""
    parts = value.split("-")
    year = parts[0]
    month = parts[1] if len(parts) > 1 else "01"
    day = parts[2] if len(parts) > 2 else "01"
    return f"{year}-{month}-{day}"


def is_pronominal(name: str) -> bool:
    """True for pronouns and relative references that cannot resolve to a vertex."""
    n = (name or "").strip().lower()
    if not n or n in PRONOUNS:
        return True
    return any(n.startswith(p) for p in PRONOMINAL_PREFIXES) and len(n.split()) <= 3


class GraphExtractor:
    """Extracts entities and triples from text.

    Prompt, entity types, predicate policy and gleaning depth are all injectable
    so callers can tune extraction per corpus without forking the library.
    """

    def __init__(
        self,
        llm_service: LLMService,
        system_prompt: Optional[str] = None,
        entity_types: Optional[Sequence[str]] = None,
        predicate_vocabulary: Optional[Sequence[str]] = None,
        predicate_aliases: Optional[Dict[str, str]] = None,
        gleaning_passes: int = 1,
        min_confidence: float = 0.0,
        drop_negated: bool = False,
        reject_possessive_entities: bool = False,
        extract_validity: bool = True,
    ):
        self.llm_service = llm_service
        self._system_prompt = system_prompt
        self.entity_types = list(entity_types) if entity_types else list(DEFAULT_ENTITY_TYPES)
        self.predicate_vocabulary = [normalise_predicate(p) for p in (predicate_vocabulary or [])]
        self.predicate_aliases = {
            normalise_predicate(k): normalise_predicate(v)
            for k, v in (predicate_aliases or {}).items()
        }
        self.gleaning_passes = max(0, gleaning_passes)
        self.min_confidence = min_confidence
        self.drop_negated = drop_negated
        self.reject_possessive_entities = reject_possessive_entities
        self.extract_validity = extract_validity

    # ------------------------------------------------------------------ prompt

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt

        types = ", ".join(f"'{t}'" for t in self.entity_types)
        type_guidance = (
            f"choose the single best fit from this list: {types}. "
            f"Use 'Concept' only when nothing else fits."
        )

        if self.predicate_vocabulary:
            vocab = ", ".join(f"'{p}'" for p in self.predicate_vocabulary)
            predicate_guidance = (
                f"  PREFER these predicates whenever one of them fits: {vocab}.\n"
                f"  Only invent a new predicate when none of the above expresses the relation."
            )
        else:
            predicate_guidance = (
                "  Reuse the same predicate wording for the same kind of relation across the whole text."
            )

        if self.extract_validity:
            validity_guidance = (
                "- valid_from / valid_to: fill these ONLY when the text states when the\n"
                "  relation began or ended ('from 1625', 'until his death in 1673',\n"
                "  'between 1628 and 1630'). Use ISO-8601 or a bare year.\n"
                "  Leave BOTH null when the text gives no period. Never guess, never infer\n"
                "  a date from context, and never use the document's own date as a default.\n"
                "  A relation with no stated period is treated as always valid, which is the\n"
                "  correct reading — do not invent one to fill the field."
            )
        else:
            validity_guidance = "- Do not populate valid_from or valid_to."

        return BASE_SYSTEM_PROMPT.format(
            type_guidance=type_guidance,
            predicate_guidance=predicate_guidance,
            validity_guidance=validity_guidance,
        )

    @staticmethod
    def _context_block(context: Optional[DocumentContext]) -> str:
        """Render document context so pronouns in a chunk can be resolved.

        Without this each chunk is extracted blind, which is how a vertex named
        'his father' ends up in the graph.
        """
        if not context:
            return ""
        parts = []
        if context.title:
            parts.append(f"Document title: {context.title}")
        if context.source:
            parts.append(f"Source: {context.source}")
        if context.summary:
            parts.append(f"Story so far: {context.summary}")
        if context.known_entities:
            names = ", ".join(context.known_entities[:40])
            parts.append(
                f"Entities already identified in this document (reuse these exact "
                f"canonical names when the text refers to them): {names}"
            )
        if not parts:
            return ""
        return "---Document Context---\n" + "\n".join(parts) + "\n\n"

    # -------------------------------------------------------------- extraction

    async def extract_from_text(
        self,
        text: str,
        context: Optional[DocumentContext] = None,
    ) -> ExtractionResult:
        """Extract entities and triples from text using the LLM.

        Raises :class:`ExtractionError` if the LLM produces nothing usable.

        There is deliberately no heuristic fallback. Placeholder edges are
        indistinguishable from genuine extracted structure once written, so a
        transient LLM outage would permanently poison the graph.
        """
        ctx = self._context_block(context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{ctx}---Document Text---\n\n{text}"},
        ]

        result = await self.llm_service.chat_completion(messages, response_format=ExtractionResult)
        if isinstance(result, str):
            result = self._parse_json_result(result)

        if not isinstance(result, ExtractionResult) or not (result.entities or result.triples):
            raise ExtractionError(
                "LLM returned no usable entities or triples. Refusing to write placeholder "
                "structure into the graph."
            )

        # Gleaning: re-prompt for what the first pass missed. Single-pass
        # extraction reliably under-recalls on dense text.
        for pass_no in range(self.gleaning_passes):
            extra = await self._glean(text, ctx, result, pass_no)
            if extra is None or not (extra.entities or extra.triples):
                break
            result = self._merge(result, extra)

        return self._validate(result)

    async def _glean(
        self, text: str, ctx: str, so_far: ExtractionResult, pass_no: int
    ) -> Optional[ExtractionResult]:
        """Ask for entities and relations missed by earlier passes."""
        entity_names = [e.name for e in so_far.entities]
        triple_strs = [f"{t.subject} -{t.predicate}-> {t.object}" for t in so_far.triples]
        glean_msg = GLEANING_PROMPT.format(
            entities=", ".join(entity_names) or "(none)",
            triples="; ".join(triple_strs) or "(none)",
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{ctx}---Document Text---\n\n{text}"},
            {"role": "user", "content": glean_msg},
        ]
        try:
            extra = await self.llm_service.chat_completion(messages, response_format=ExtractionResult)
            if isinstance(extra, str):
                extra = self._parse_json_result(extra)
            return extra if isinstance(extra, ExtractionResult) else None
        except Exception as e:
            # Gleaning is an enhancement; the first pass already succeeded.
            logger.warning("Gleaning pass %d failed (%s); keeping earlier results.", pass_no + 1, e)
            return None

    @staticmethod
    def _merge(base: ExtractionResult, extra: ExtractionResult) -> ExtractionResult:
        """Union two extractions, preferring the richer record for duplicates."""
        entities: Dict[str, Entity] = {}
        for e in [*base.entities, *extra.entities]:
            key = e.name.strip().lower()
            if not key:
                continue
            prior = entities.get(key)
            if prior is None:
                entities[key] = e
                continue
            merged_aliases = list(dict.fromkeys([*prior.aliases, *e.aliases]))
            entities[key] = Entity(
                name=prior.name,
                type=prior.type if prior.type != "Concept" else e.type,
                description=prior.description or e.description,
                aliases=merged_aliases,
            )

        triples: Dict[tuple, Triple] = {}
        for t in [*base.triples, *extra.triples]:
            key = (t.subject.strip().lower(), normalise_predicate(t.predicate), t.object.strip().lower())
            prior = triples.get(key)
            if prior is None:
                triples[key] = t
                continue
            # Keep whichever pass supplied a stated period; a gleaning pass often
            # surfaces the dates the first pass omitted.
            if prior.valid_from is None and t.valid_from is not None:
                prior.valid_from = t.valid_from
            if prior.valid_to is None and t.valid_to is not None:
                prior.valid_to = t.valid_to

        return ExtractionResult(entities=list(entities.values()), triples=list(triples.values()))

    @staticmethod
    def _parse_json_result(raw: str) -> Optional[ExtractionResult]:
        """Parse a JSON extraction payload, tolerating markdown code fences."""
        if not raw or not raw.strip():
            return None
        cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return ExtractionResult(**json.loads(cleaned))
        except Exception as e:
            logger.debug("Could not parse extraction JSON: %s", e)
            return None

    def _canonical_predicate(self, predicate: str) -> str:
        """Normalise, then snap onto the configured vocabulary where possible."""
        p = normalise_predicate(predicate)
        p = self.predicate_aliases.get(p, p)
        if self.predicate_vocabulary and p not in self.predicate_vocabulary:
            # Accept a vocabulary entry that is a clear morphological match.
            for candidate in self.predicate_vocabulary:
                if p.startswith(candidate) or candidate.startswith(p):
                    return candidate
        return p

    def _validate(self, result: ExtractionResult) -> ExtractionResult:
        """Drop malformed, pronominal and semantically empty records.

        Filters vague predicates, self-loops and unresolved references so that a
        relation surviving into the graph always says something specific about a
        pair of entities that can actually be identified.
        """
        entities, dropped_names = [], set()
        for e in result.entities:
            name = (e.name or "").strip()
            if not name or is_pronominal(name) or is_phrase_not_entity(name, self.reject_possessive_entities):
                if name:
                    dropped_names.add(name.lower())
                    logger.debug("Dropping non-entity name: %s", name)
                continue
            aliases = [a.strip() for a in e.aliases
                       if a and a.strip() and not is_pronominal(a)
                       and len(a.strip()) <= MAX_ENTITY_NAME_CHARS]
            aliases = [a for a in dict.fromkeys(aliases) if a.lower() != name.lower()]
            entities.append(Entity(name=name, type=e.type or "Concept",
                                   description=e.description or "", aliases=aliases))

        triples = []
        for t in result.triples:
            subject = (t.subject or "").strip()
            obj = (t.object or "").strip()
            predicate = self._canonical_predicate(t.predicate)

            if not subject or not obj or not predicate:
                continue
            if predicate in VAGUE_PREDICATES:
                logger.debug("Dropping vague triple: (%s)-[%s]->(%s)", subject, predicate, obj)
                continue
            if subject.lower() == obj.lower():
                continue
            if is_pronominal(subject) or is_pronominal(obj):
                continue
            if (is_phrase_not_entity(subject, self.reject_possessive_entities)
                    or is_phrase_not_entity(obj, self.reject_possessive_entities)):
                continue
            if subject.lower() in dropped_names or obj.lower() in dropped_names:
                continue
            if t.confidence < self.min_confidence:
                continue
            if t.negated and self.drop_negated:
                continue

            triples.append(Triple(
                subject=subject, predicate=predicate, object=obj,
                description=t.description, negated=bool(t.negated),
                confidence=float(t.confidence),
                # Absent validity stays absent. A relation with no stated period
                # is always valid, and inventing a period here would silently
                # narrow it.
                valid_from=_clean_date(t.valid_from) if self.extract_validity else None,
                valid_to=_clean_date(t.valid_to) if self.extract_validity else None,
            ))

        if not entities and not triples:
            raise ExtractionError(
                "All extracted entities and triples were rejected as empty, pronominal or non-specific."
            )

        return ExtractionResult(entities=entities, triples=triples)

    # ---------------------------------------------------------------- keywords

    async def detect_contradictions(
        self,
        new_fact: str,
        candidates: List[Dict[str, Any]],
    ) -> List[str]:
        """Which existing facts does the new one make false?

        Returns candidate IDs. An empty list on any failure — a wrong answer
        here retracts a true fact from every future query, so the failure mode
        has to be "changed nothing", not "guessed". That is the opposite of the
        fail-closed rule used during extraction, and deliberately so: there the
        risk is silently indexing nothing, here it is silently deleting
        something.

        Only IDs that were actually offered are returned. A model that invents
        an ID would otherwise retract an unrelated relation.
        """
        if not candidates:
            return []
        offered = {str(c["id"]) for c in candidates}
        listing = "\n".join(
            f"- ID {c['id']}: {c.get('description') or c.get('relation_type')}"
            + (f" [valid {c.get('valid_from') or '?'} to {c.get('valid_to') or 'present'}]"
               if c.get("valid_from") or c.get("valid_to") else "")
            for c in candidates
        )
        messages = [
            {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"NEW FACT:\n{new_fact}\n\nEXISTING FACTS:\n{listing}"},
        ]
        try:
            res = await self.llm_service.chat_completion(
                messages, response_format=ContradictionResultSchema)
            if isinstance(res, ContradictionResultSchema):
                ids = res.contradicted_ids
            elif isinstance(res, str) and res.strip():
                cleaned = res.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
                ids = json.loads(cleaned).get("contradicted_ids", [])
            else:
                return []
        except Exception as e:
            logger.warning("Contradiction detection failed (%s); retracting nothing.", e)
            return []
        return [str(i) for i in ids if str(i) in offered]

    async def extract_keywords(self, query: str) -> KeywordResult:
        """Extract high-level and low-level keywords from a user query.

        Unlike extraction, a weak keyword list is not persisted and only widens
        retrieval, so a lexical fallback is acceptable here.
        """
        messages = [
            {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Query: {query}"},
        ]
        try:
            res = await self.llm_service.chat_completion(messages, response_format=KeywordResultSchema)
            if isinstance(res, KeywordResultSchema):
                return KeywordResult(
                    high_level_keywords=res.high_level_keywords,
                    low_level_keywords=res.low_level_keywords,
                )
            if isinstance(res, str) and res.strip():
                data = json.loads(res)
                return KeywordResult(
                    high_level_keywords=data.get("high_level_keywords", []),
                    low_level_keywords=data.get("low_level_keywords", []),
                )
        except Exception as e:
            logger.warning("Keyword extraction failed (%s); falling back to lexical split.", e)

        words = [w.strip("?,.!") for w in query.split() if len(w) > 2]
        return KeywordResult(high_level_keywords=[query], low_level_keywords=words)
