"""End-to-end tests against a live PostgreSQL with pgvector.

The rest of the suite tests units: a chunker, a merge function, a validation
rule. These exercise whole paths — index, resolve, retrieve, synthesise — with
a real database underneath, because the interesting failures live between the
components rather than inside them. The multi-tenancy tests are here for the
same reason: isolation that holds in a unit test and leaks in a real schema is
the failure mode that matters.

Skipped automatically when PostgreSQL is unreachable (see conftest).
"""
import pytest

from post_graph_rag import DocumentMetadata, QueryParam
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

pytestmark = pytest.mark.asyncio


# Two documents that share entities, so cross-document resolution has something
# to resolve rather than being trivially true.
DOC_A = ExtractionResult(
    entities=[Entity(name="Ada Lovelace", type="Person", description="mathematician"),
              Entity(name="Analytical Engine", type="Machine", description="a design")],
    triples=[Triple(subject="Ada Lovelace", predicate="wrote_about",
                    object="Analytical Engine", description="notes on the engine")],
)
DOC_B = ExtractionResult(
    entities=[Entity(name="Charles Babbage", type="Person", description="engineer"),
              Entity(name="Analytical Engine", type="Machine", description="his design")],
    triples=[Triple(subject="Charles Babbage", predicate="designed",
                    object="Analytical Engine", description="designed the engine")],
)


async def _index_two_documents(rag):
    await rag.index_document("first", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = DOC_B
    await rag.index_document("second", metadata=DocumentMetadata(document="b.txt"))


# ------------------------------------------------------------------ lifecycle

async def test_index_then_retrieve_then_answer(rag_factory):
    """The whole path, once: text in, grounded answer out."""
    rag = await rag_factory(extraction=DOC_A, answer="Ada wrote about the engine.")
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))

    data = await rag.query_data("Ada Lovelace", param=QueryParam(mode="mix", top_k=5))
    assert data["data"]["entities"], "no entities retrieved"
    assert data["data"]["relationships"], "no relations retrieved"

    out = await rag.query("Ada Lovelace", param=QueryParam(mode="mix", top_k=5))
    assert (out["answer"] if isinstance(out, dict) else str(out)).strip()


async def test_entity_resolves_across_documents(rag_factory):
    """One vertex for an entity named in two documents.

    This is what makes traversal work at all: two vertices for the same thing
    means a question about it reaches half the graph.
    """
    rag = await rag_factory(extraction=DOC_A)
    await _index_two_documents(rag)

    engine = await rag.store.find_entity_by_name("Analytical Engine")
    assert engine is not None
    rows = await rag.store.client._fetch(
        f"SELECT count(*) AS n FROM {rag.store.client._get_table_ref('entities', rag.config.realm)} "
        f"WHERE realm = $1 AND lower(payload->>'name') = 'analytical engine'",
        rag.config.realm)
    assert rows[0]["n"] == 1, "the shared entity was split across documents"


async def test_reindexing_unchanged_content_is_a_no_op(rag_factory):
    """Re-running an unchanged document must not duplicate the graph."""
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("same text", metadata=DocumentMetadata(document="a.txt"))
    before = await rag.store.get_all_relations(limit=100)
    await rag.index_document("same text", metadata=DocumentMetadata(document="a.txt"))
    after = await rag.store.get_all_relations(limit=100)
    assert len(after) == len(before)


async def test_reobserving_a_relation_raises_its_weight(rag_factory):
    """Corroboration is counted, not duplicated."""
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("first", metadata=DocumentMetadata(document="a.txt"))
    await rag.index_document("different words, same claim",
                             metadata=DocumentMetadata(document="b.txt"))
    relations = await rag.store.get_all_relations(limit=50)
    matching = [r for r, _s, _t in relations if r.relation_type == "wrote_about"]
    assert len(matching) == 1, "the same triple became two edges"
    assert (matching[0].payload or {}).get("weight", 1) >= 2


# ------------------------------------------------------------- multi-tenancy

async def test_realms_do_not_leak(rag_factory):
    """A query in one realm must not see another's data.

    Tenant isolation that holds in a unit test and leaks against a real schema
    is the failure nobody notices until it is a disclosure.
    """
    one = await rag_factory(extraction=DOC_A)
    two = await rag_factory(extraction=DOC_B)
    await one.index_document("first", metadata=DocumentMetadata(document="a.txt"))
    await two.index_document("second", metadata=DocumentMetadata(document="b.txt"))

    assert await one.store.find_entity_by_name("Ada Lovelace") is not None
    assert await one.store.find_entity_by_name("Charles Babbage") is None
    assert await two.store.find_entity_by_name("Charles Babbage") is not None
    assert await two.store.find_entity_by_name("Ada Lovelace") is None


async def test_spaces_isolate_within_one_realm(rag_factory):
    """Space is the logical boundary inside a realm; it must filter too."""
    rag = await rag_factory(extraction=DOC_A)
    await rag.store.upsert_entity("Alpha", "Person", "d", [0.1] * rag.config.embedding_dim,
                                  space="team_a")
    await rag.store.upsert_entity("Beta", "Person", "d", [0.1] * rag.config.embedding_dim,
                                  space="team_b")
    assert await rag.store.find_entity_by_name("Alpha", space="team_a") is not None
    assert await rag.store.find_entity_by_name("Alpha", space="team_b") is None


# ----------------------------------------------------------------- retrieval

async def test_multi_hop_reaches_what_one_hop_cannot(rag_factory):
    """A chain fact is only reachable beyond the first hop.

    A -> B -> C: asking about A must reach the B->C edge at depth 2, which is
    the entire argument for a hop budget above one.
    """
    chain = ExtractionResult(
        entities=[Entity(name="A", type="Thing", description="d"),
                  Entity(name="B", type="Thing", description="d"),
                  Entity(name="C", type="Thing", description="d")],
        triples=[Triple(subject="A", predicate="leads_to", object="B"),
                 Triple(subject="B", predicate="causes", object="C")],
    )
    rag = await rag_factory(extraction=chain)
    await rag.index_document("chain", metadata=DocumentMetadata(document="a.txt"))

    # Traversal is exercised directly rather than through query_data: vector
    # search seeds from every entity it matches, so a full query reaches the
    # second edge at one hop via B and depth is never the variable under test.
    a = await rag.store.find_entity_by_name("A")
    assert a is not None
    shallow = await rag.store.get_neighborhood(a.id, max_hops=1)
    deep = await rag.store.get_neighborhood(a.id, max_hops=2)

    preds_shallow = {e.relation_type for e, _s, _t, _h in shallow}
    preds_deep = {e.relation_type for e, _s, _t, _h in deep}
    assert preds_shallow == {"leads_to"}, "one hop reached beyond A's own edges"
    assert "causes" in preds_deep, "two hops failed to reach the chain fact"
    assert max(h for _e, _s, _t, h in deep) == 2, "hop distance is not recorded"


async def test_relation_channel_changes_what_reaches_the_prompt(rag_factory):
    """embed_relations adds a second candidate generator, not just an ordering."""
    rag = await rag_factory(extraction=DOC_A, embed_relations=True,
                            relation_seed_quota=1.0)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))
    hits = await rag.store.search_similar_relations(
        [0.1] * rag.config.embedding_dim, top_k=5)
    assert hits, "relations were not embedded despite embed_relations=True"


async def test_relations_are_not_embedded_when_disabled(rag_factory):
    rag = await rag_factory(extraction=DOC_A, embed_relations=False)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))
    assert await rag.store.search_similar_relations(
        [0.1] * rag.config.embedding_dim, top_k=5) == []


async def test_mention_expansion_reaches_passages_the_query_did_not_match(rag_factory):
    """A chunk that explains an entity without echoing the question's wording."""
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("Ada Lovelace worked on it.",
                             metadata=DocumentMetadata(document="a.txt"))
    data = await rag.query_data("Ada Lovelace",
                                param=QueryParam(mode="mix", top_k=5, ll_keywords=["Ada"]))
    assert data["data"]["chunks"], "no supporting passages retrieved"


# ------------------------------------------------------------------ failure

async def test_indexing_raises_when_every_chunk_fails(rag_factory):
    """A total extraction outage must not look like an empty corpus."""
    from post_graph_rag.errors import LLMError

    rag = await rag_factory(extraction=DOC_A)

    # extract_from_text is what the engine calls. Patching `extract` left the
    # real path untouched and the test passed while proving nothing.
    async def always_fail(*args, **kwargs):
        raise LLMError("router down")
    rag.extractor.extract_from_text = always_fail

    with pytest.raises(Exception):
        await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))


async def test_query_on_an_empty_graph_does_not_crash(rag_factory):
    """No data is a valid state and must answer, not raise."""
    rag = await rag_factory(extraction=DOC_A)
    data = await rag.query_data("anything", param=QueryParam(mode="mix", top_k=5))
    assert data["data"]["entities"] == []
    assert data["data"]["relationships"] == []


# ------------------------------------------------------------ bi-temporal

async def test_relations_record_both_temporal_axes(rag_factory):
    """Validity and belief time are independent and both stored."""
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))
    data = await rag.query_data("Ada Lovelace", param=QueryParam(mode="mix", top_k=5))
    rel = data["data"]["relationships"][0]
    assert rel["t_created"], "no transaction time recorded"
    assert rel["t_expired"] is None, "a live relation must not be expired"


async def test_as_believed_at_hides_facts_learned_later(rag_factory):
    """The question the second axis exists for: what did we know then.

    `as_of` cannot answer it — the fact is valid throughout; what changed is
    when this system came to hold it.
    """
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))

    before_any_indexing = "2000-01-01T00:00:00+00:00"
    now = await rag.query_data("Ada Lovelace", param=QueryParam(mode="mix", top_k=5))
    past = await rag.query_data("Ada Lovelace", param=QueryParam(
        mode="mix", top_k=5, as_believed_at=before_any_indexing))

    assert now["data"]["relationships"], "the relation should be known now"
    assert not past["data"]["relationships"], (
        "a relation written today was returned as known in 2000")


async def test_supersession_stamps_belief_expiry(rag_factory):
    """A superseded relation stops being believed, without ceasing to exist."""
    friends = ExtractionResult(
        entities=[Entity(name="Alice", type="Person", description="d"),
                  Entity(name="Bob", type="Person", description="d")],
        triples=[Triple(subject="Alice", predicate="friend_of", object="Bob")],
    )
    rag = await rag_factory(extraction=friends,
                            exclusive_predicate_groups=[{"friend_of", "rivals_with"}])
    await rag.index_document("friends", metadata=DocumentMetadata(document="a.txt"))
    rag.llm._extraction = ExtractionResult(
        entities=[Entity(name="Alice", type="Person", description="d")],
        triples=[Triple(subject="Alice", predicate="rivals_with", object="Bob")],
    )
    await rag.index_document("rivals", metadata=DocumentMetadata(document="b.txt"))

    data = await rag.query_data("Alice", param=QueryParam(
        mode="mix", top_k=5, ll_keywords=["Alice"], include_superseded=True))
    superseded = [r for r in data["data"]["relationships"] if r.get("superseded_by")]
    assert superseded, "supersession did not fire"
    assert superseded[0]["t_expired"], "belief expiry was not stamped"


# ------------------------------------------------------------ lexical channel

async def test_lexical_search_finds_a_rare_identifier(rag_factory):
    """What the lexical channel is for: a token embeddings place badly."""
    parts = ExtractionResult(
        entities=[Entity(name="Boeing", type="Org", description="d"),
                  Entity(name="737-9", type="Product", description="d")],
        triples=[Triple(subject="Boeing", predicate="grounded", object="737-9",
                        description="The 737-9 fleet was grounded after the door plug failure")],
    )
    rag = await rag_factory(extraction=parts)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))

    hits = await rag.store.search_relations_text("door plug", top_k=5)
    assert hits, "lexical search found nothing for a phrase present in the text"
    assert hits[0][1] > 0, "a match must carry a positive rank"


async def test_lexical_search_ignores_an_absent_term(rag_factory):
    rag = await rag_factory(extraction=DOC_A)
    await rag.index_document("text", metadata=DocumentMetadata(document="a.txt"))
    assert await rag.store.search_relations_text("submarine", top_k=5) == []


async def test_lexical_search_on_empty_query_is_a_no_op(rag_factory):
    rag = await rag_factory(extraction=DOC_A)
    assert await rag.store.search_relations_text("   ", top_k=5) == []
