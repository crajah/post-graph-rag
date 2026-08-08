"""Tests for the extraction-quality fixes: aliases, context, negation, gleaning, vocabulary."""
import pytest

from post_graph_rag import RAGConfig
from post_graph_rag.chunking import paragraph_chunker
from post_graph_rag.extractor import (
    MAX_ENTITY_NAME_CHARS,
    Entity, ExtractionResult, GraphExtractor, Triple,
    is_phrase_not_entity, is_pronominal, normalise_predicate,
)
from post_graph_rag.models import DocumentContext

from conftest import FakeLLM


def _extractor(extraction=None, **kw):
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    kw.setdefault("gleaning_passes", 0)
    return GraphExtractor(FakeLLM(config, extraction=extraction), **kw)


# ------------------------------------------------------------------- aliases

@pytest.mark.asyncio
async def test_aliases_are_captured_and_cleaned():
    result = ExtractionResult(
        entities=[Entity(name="Charles Babbage", type="Person", description="inventor",
                         aliases=["Babbage", "Charles Babbage", "he", "his father"])],
        triples=[Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine")],
    )
    out = await _extractor(result).extract_from_text("text")
    aliases = out.entities[0].aliases
    assert "Babbage" in aliases
    assert "Charles Babbage" not in aliases   # the canonical name is not its own alias
    assert "he" not in aliases                # pronouns are not surface forms worth storing
    assert "his father" not in aliases


# ----------------------------------------------------------------- pronouns

def test_is_pronominal():
    for bad in ["he", "She", "they", "his father", "the incident", "their company", "The author"]:
        assert is_pronominal(bad), bad
    for good in ["Charles Babbage", "The Analytical Society", "The Times", "Analytical Engine"]:
        assert not is_pronominal(good), good


@pytest.mark.asyncio
async def test_pronominal_entities_and_their_triples_are_dropped():
    """'his father' as a vertex can never resolve or connect to anything."""
    result = ExtractionResult(
        entities=[
            Entity(name="Charles Babbage", type="Person", description="inventor"),
            Entity(name="his father", type="Person", description="a relative"),
        ],
        triples=[
            Triple(subject="Charles Babbage", predicate="son_of", object="his father"),
            Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine"),
        ],
    )
    out = await _extractor(result).extract_from_text("text")
    assert [e.name for e in out.entities] == ["Charles Babbage"]
    assert [t.predicate for t in out.triples] == ["designed"]


def test_conjunction_entities_rejected_by_default():
    """'X and Y' is two entities plus a relation, never one entity."""
    for bad in ["Ada Lovelace and Charles Babbage", "Babbage and Herschel",
                "Larcom Street and Walworth Road"]:
        assert is_phrase_not_entity(bad), bad
    # Real names containing 'and'/'of'/possessives must survive.
    for good in ["The Thrilling Adventures of Lovelace and Babbage",
                 "Astronomical Society of London", "Babbage's Analytical Engine",
                 "Barons Lovelace", "Charles Babbage", "Analytical Engine"]:
        assert not is_phrase_not_entity(good), good


def test_possessive_rejection_is_opt_in():
    """Off by default because it also catches legitimately named things."""
    for phrase in ["Babbage's father", "Ada Lovelace's affair"]:
        assert not is_phrase_not_entity(phrase)
        assert is_phrase_not_entity(phrase, reject_possessive=True)
    # The cost of enabling it: these are real named things, and are lost too.
    for casualty in ["Ampere's force law", "Menabrea's paper"]:
        assert is_phrase_not_entity(casualty, reject_possessive=True)


@pytest.mark.asyncio
async def test_phrase_entities_and_their_triples_dropped():
    result = ExtractionResult(
        entities=[
            Entity(name="Charles Babbage", type="Person", description="inventor"),
            Entity(name="Ada Lovelace and Charles Babbage", type="Person", description="pair"),
        ],
        triples=[
            Triple(subject="Ada Lovelace and Charles Babbage", predicate="worked_on", object="Engine"),
            Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine"),
        ],
    )
    out = await _extractor(result).extract_from_text("text")
    assert [e.name for e in out.entities] == ["Charles Babbage"]
    assert [t.predicate for t in out.triples] == ["designed"]


# ------------------------------------------------------------------ context

@pytest.mark.asyncio
async def test_document_context_is_sent_to_the_model():
    ex = _extractor(ExtractionResult(
        entities=[Entity(name="Ada Lovelace", type="Person", description="d")],
        triples=[],
    ))
    ctx = DocumentContext(
        title="Charles Babbage",
        source="https://example.org/babbage",
        summary="Babbage designed calculating engines.",
        known_entities=["Charles Babbage", "Analytical Engine"],
    )
    await ex.extract_from_text("He then designed the engine.", context=ctx)
    sent = ex.llm_service.chat_calls[0][0][-1]["content"]
    assert "Charles Babbage" in sent
    assert "https://example.org/babbage" in sent
    assert "Babbage designed calculating engines." in sent
    assert "Analytical Engine" in sent


@pytest.mark.asyncio
async def test_no_context_block_when_context_absent():
    ex = _extractor(ExtractionResult(entities=[Entity(name="X", type="Concept", description="d")]))
    await ex.extract_from_text("some text")
    assert "---Document Context---" not in ex.llm_service.chat_calls[0][0][-1]["content"]


# ----------------------------------------------------------------- negation

@pytest.mark.asyncio
async def test_negated_relation_is_flagged_not_inverted():
    result = ExtractionResult(
        entities=[Entity(name="Ada Lovelace", type="Person", description="d")],
        triples=[Triple(subject="Ada Lovelace", predicate="had_relationship_with",
                        object="Lord Byron", negated=True)],
    )
    out = await _extractor(result).extract_from_text("text")
    assert out.triples[0].negated is True
    assert "not" not in out.triples[0].predicate


@pytest.mark.asyncio
async def test_negated_relations_can_be_dropped():
    result = ExtractionResult(
        entities=[Entity(name="Ada Lovelace", type="Person", description="d")],
        triples=[
            Triple(subject="Ada Lovelace", predicate="had_relationship_with", object="Lord Byron", negated=True),
            Triple(subject="Ada Lovelace", predicate="worked_with", object="Charles Babbage"),
        ],
    )
    out = await _extractor(result, drop_negated=True).extract_from_text("text")
    assert [t.predicate for t in out.triples] == ["worked_with"]


# --------------------------------------------------------------- confidence

@pytest.mark.asyncio
async def test_low_confidence_relations_filtered():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Concept", description="d")],
        triples=[
            Triple(subject="A", predicate="maybe_causes", object="B", confidence=0.2),
            Triple(subject="A", predicate="causes", object="C", confidence=0.9),
        ],
    )
    out = await _extractor(result, min_confidence=0.5).extract_from_text("text")
    assert [t.predicate for t in out.triples] == ["causes"]


# --------------------------------------------------------------- vocabulary

def test_normalise_predicate():
    assert normalise_predicate("Was Appointed Knight-Of") == "appointed_knight_of"
    assert normalise_predicate("  collaborated   with ") == "collaborated_with"
    assert normalise_predicate("is_a") == "is_a"          # too short to strip
    assert normalise_predicate("HAS_COMPONENT") == "component"


@pytest.mark.asyncio
async def test_predicate_vocabulary_snaps_predicates():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Concept", description="d")],
        triples=[
            Triple(subject="A", predicate="worked_with_closely", object="B"),
            Triple(subject="A", predicate="designed", object="C"),
        ],
    )
    ex = _extractor(result, predicate_vocabulary=["worked_with", "designed"])
    out = await ex.extract_from_text("text")
    assert sorted(t.predicate for t in out.triples) == ["designed", "worked_with"]


@pytest.mark.asyncio
async def test_predicate_aliases_applied():
    result = ExtractionResult(
        entities=[Entity(name="A", type="Concept", description="d")],
        triples=[Triple(subject="A", predicate="collaborated_with", object="B")],
    )
    ex = _extractor(result, predicate_aliases={"collaborated_with": "worked_with"})
    out = await ex.extract_from_text("text")
    assert out.triples[0].predicate == "worked_with"


@pytest.mark.asyncio
async def test_vocabulary_appears_in_prompt():
    ex = _extractor(ExtractionResult(entities=[Entity(name="A", type="Concept", description="d")]),
                    predicate_vocabulary=["worked_with"], entity_types=["Person", "Machine"])
    assert "'worked_with'" in ex.system_prompt
    assert "'Machine'" in ex.system_prompt


@pytest.mark.asyncio
async def test_custom_system_prompt_overrides():
    ex = _extractor(ExtractionResult(entities=[Entity(name="A", type="Concept", description="d")]),
                    system_prompt="CUSTOM PROMPT")
    assert ex.system_prompt == "CUSTOM PROMPT"


# ----------------------------------------------------------------- gleaning

@pytest.mark.asyncio
async def test_gleaning_merges_missed_records():
    first = ExtractionResult(
        entities=[Entity(name="Charles Babbage", type="Person", description="inventor")],
        triples=[Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine")],
    )
    second = ExtractionResult(
        entities=[Entity(name="Ada Lovelace", type="Person", description="mathematician")],
        triples=[Triple(subject="Ada Lovelace", predicate="worked_with", object="Charles Babbage")],
    )
    ex = _extractor([first, second], gleaning_passes=1)
    out = await ex.extract_from_text("text")
    assert sorted(e.name for e in out.entities) == ["Ada Lovelace", "Charles Babbage"]
    assert len(out.triples) == 2


@pytest.mark.asyncio
async def test_gleaning_does_not_duplicate():
    same = ExtractionResult(
        entities=[Entity(name="Charles Babbage", type="Person", description="inventor")],
        triples=[Triple(subject="Charles Babbage", predicate="designed", object="Analytical Engine")],
    )
    ex = _extractor(same, gleaning_passes=2)
    out = await ex.extract_from_text("text")
    assert len(out.entities) == 1
    assert len(out.triples) == 1


@pytest.mark.asyncio
async def test_gleaning_disabled_makes_one_call():
    result = ExtractionResult(entities=[Entity(name="A", type="Concept", description="d")])
    ex = _extractor(result, gleaning_passes=0)
    await ex.extract_from_text("text")
    assert len(ex.llm_service.chat_calls) == 1


# ----------------------------------------------------------------- chunking

def test_paragraph_chunker_overlaps():
    text = "\n".join(f"Paragraph {i} with enough text to exceed the minimum length threshold." for i in range(30))
    chunks = paragraph_chunker(text, chunk_chars=300, overlap_chars=80)
    assert len(chunks) > 1
    # Consecutive chunks must share text, or a relation spanning the boundary is
    # invisible to both.
    assert any(chunks[0][-40:].split()[-1] in chunks[1] for _ in [0])


def test_paragraph_chunker_skips_headings_and_stubs():
    text = "== History ==\nshort\n" + "A properly long paragraph that should certainly be kept in full."
    chunks = paragraph_chunker(text, chunk_chars=500, overlap_chars=0)
    assert len(chunks) == 1
    assert "== History ==" not in chunks[0]
    assert "short" not in chunks[0].split("\n")


def test_oversized_name_rejected_before_it_reaches_the_index():
    """A whole table row handed back as an entity must not reach the database.

    Entity resolution relies on a unique btree index over the name, and Postgres
    rejects index keys beyond ~2704 bytes — a real SEC filing run aborted this
    way. The guard belongs in extraction, not in error handling at the store.
    """
    row = "Deferred production costs " + "and unamortized tooling " * 200
    assert len(row) > 2704
    assert is_phrase_not_entity(row) is True
    assert is_phrase_not_entity("Deferred Production Costs") is False


def test_oversized_alias_dropped_but_entity_kept():
    """One runaway alias should cost the alias, not the entity."""
    result = ExtractionResult(
        entities=[Entity(name="Boeing", type="Org", description="",
                         aliases=["BA", "x" * (MAX_ENTITY_NAME_CHARS + 1)])],
        triples=[],
    )
    cleaned = _extractor()._validate(result)
    assert [e.name for e in cleaned.entities] == ["Boeing"]
    assert cleaned.entities[0].aliases == ["BA"]


@pytest.mark.parametrize("figure", [
    "$1,326", "$18.4 billion", "$177", "98", "40%", "(1,204)", "$ 2.5 million", "12.7 percent",
])
def test_bare_quantities_are_not_entities(figure):
    """Figures fragment the graph and block supersession.

    Every annual filing reports different numbers, so a relation whose object is
    a figure never recurs as the same pair of vertices — and supersession only
    fires on a repeated pair. Observed on real 10-Ks: "Boeing -> $1,326".
    """
    assert is_phrase_not_entity(figure) is True


@pytest.mark.parametrize("name", [
    "717 aircraft", "$177 tax benefit", "737 MAX", "Boeing Company",
    "3M", "Section 401(k)",
])
def test_names_containing_figures_are_kept(name):
    """The guard must stay narrow: a figure inside a name is not a bare figure."""
    assert is_phrase_not_entity(name) is False
