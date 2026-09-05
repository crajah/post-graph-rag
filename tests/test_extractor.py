"""Tests for GraphExtractor: no fabricated relations, specific predicates only."""
import pytest
from conftest import FakeLLM

from post_graph_rag import RAGConfig
from post_graph_rag.errors import ExtractionError, RAGError
from post_graph_rag.extractor import Entity, ExtractionResult, GraphExtractor, Triple


def _extractor(extraction=None, fail=False):
    config = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
    return GraphExtractor(FakeLLM(config, extraction=extraction, fail=fail))


@pytest.mark.asyncio
async def test_extracts_specific_relations():
    result = ExtractionResult(
        entities=[Entity(name="Zeus", type="Person", description="king of the gods")],
        triples=[Triple(subject="Zeus", predicate="married_to", object="Hera", description="spouse")],
    )
    out = await _extractor(result).extract_from_text("Zeus is married to Hera.")
    assert [t.predicate for t in out.triples] == ["married_to"]


@pytest.mark.asyncio
async def test_llm_failure_raises_instead_of_fabricating_generic_edges():
    """The core regression.

    The old fallback chained adjacent capitalised words with 'relates_to',
    writing edges like (Mount)-[relates_to]->(Olympus) into the graph while
    reporting success. A transient outage permanently poisoned the graph.

    Transport failures surface as LLMError and unusable output as
    ExtractionError; what matters is that neither is swallowed into fabricated
    structure, so this asserts on the shared RAGError base.
    """
    text = (
        "Zeus is the king of the Olympian gods, ruling from Mount Olympus. "
        "He is the son of Cronus and Rhea, and married to Hera."
    )
    with pytest.raises(RAGError):
        await _extractor(fail=True).extract_from_text(text)


@pytest.mark.asyncio
async def test_empty_extraction_raises():
    with pytest.raises(ExtractionError):
        await _extractor(ExtractionResult()).extract_from_text("some text")


@pytest.mark.asyncio
async def test_vague_predicates_are_dropped():
    result = ExtractionResult(
        entities=[Entity(name="Zeus", type="Person", description="d")],
        triples=[
            Triple(subject="Zeus", predicate="relates_to", object="Hera"),
            Triple(subject="Zeus", predicate="associated_with", object="Olympus"),
            Triple(subject="Zeus", predicate="married_to", object="Hera"),
        ],
    )
    out = await _extractor(result).extract_from_text("text")
    assert [t.predicate for t in out.triples] == ["married_to"]


@pytest.mark.asyncio
async def test_self_loops_and_blank_endpoints_dropped():
    result = ExtractionResult(
        entities=[Entity(name="Zeus", type="Person", description="d")],
        triples=[
            Triple(subject="Zeus", predicate="is", object="zeus"),
            Triple(subject="", predicate="rules", object="Olympus"),
            Triple(subject="Zeus", predicate="", object="Hera"),
            Triple(subject="Zeus", predicate="rules", object="Olympus"),
        ],
    )
    out = await _extractor(result).extract_from_text("text")
    assert [(t.subject, t.predicate, t.object) for t in out.triples] == [("Zeus", "rules", "Olympus")]


@pytest.mark.asyncio
async def test_predicates_are_normalised():
    """Case and separators are normalised, and leading tense auxiliaries stripped.

    'was_appointed_knight_of' and 'appointed_knight_of' describe the same
    relation; keeping both splits the edge vocabulary for no gain.
    """
    result = ExtractionResult(
        entities=[Entity(name="Zeus", type="Person", description="d")],
        triples=[
            Triple(subject="Zeus", predicate="Is King Of", object="Olympus"),
            Triple(subject="Zeus", predicate="was appointed  Knight-Of", object="Order"),
            Triple(subject="Zeus", predicate="is_a", object="Deity"),
        ],
    )
    out = await _extractor(result).extract_from_text("text")
    assert [t.predicate for t in out.triples] == ["king_of", "appointed_knight_of", "is_a"]


@pytest.mark.asyncio
async def test_all_vague_triples_and_no_entities_raises():
    result = ExtractionResult(
        entities=[],
        triples=[Triple(subject="A", predicate="relates_to", object="B")],
    )
    with pytest.raises(ExtractionError):
        await _extractor(result).extract_from_text("text")


@pytest.mark.asyncio
async def test_keyword_extraction_falls_back_lexically():
    """Keywords are not persisted and only widen retrieval, so a weak fallback
    is acceptable here even though extraction refuses to guess."""
    kw = await _extractor(fail=True).extract_keywords("Who created the Python language?")
    assert kw.high_level_keywords == ["Who created the Python language?"]
    assert "Python" in kw.low_level_keywords
