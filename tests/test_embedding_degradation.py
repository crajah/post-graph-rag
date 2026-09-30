"""The chunk's own vector is the only one its passage needs.

engine.py already argues that a chunk whose extraction fails should still be
written, because the vector channel never needed entities. An EmbeddingError on
the three *derived* batches -- entity, stub and relation embeddings -- is the
same class of event, and by then the chunk's own vector has already been
computed. Losing the passage over it is the same bug in a different coat.

Underneath that sits the cause. Endpoints cap inputs per embedding request and
exceeding the cap is a 400, which is not retryable, so an unbatched request
simply fails. A chunk that extracted more entities than the cap therefore lost
its whole passage on a limit that has nothing to do with the passage.
"""
import pytest
from conftest import fake_embed, make_config

from post_graph_rag import DocumentMetadata
from post_graph_rag.errors import EmbeddingError
from post_graph_rag.extractor import Entity, ExtractionResult, Triple
from post_graph_rag.llm import LLMService

TEXT = "Zeus is king of the Olympian gods, married to Hera."

RICH = ExtractionResult(
    entities=[
        Entity(name="Zeus", type="Person", description="king of the gods"),
        Entity(name="Hera", type="Person", description="wife of Zeus"),
    ],
    triples=[Triple(subject="Zeus", predicate="married_to", object="Hera",
                    description="spouse")],
)


class TestRequestsRespectTheEndpointLimit:
    """The cause. Without this the catch below fires on every dense chunk,
    which trades lost passages for quietly lost structure."""

    @pytest.mark.asyncio
    async def test_a_batch_over_the_limit_is_split(self):
        svc = LLMService(make_config(embedding_batch_size=64))
        sizes = []

        async def record(texts):
            sizes.append(len(texts))
            return [fake_embed(t) for t in texts]

        svc._embed_batch = record
        out = await svc.get_embeddings([f"e{i}" for i in range(150)])

        assert sizes == [64, 64, 22], f"expected three capped requests, got {sizes}"
        assert len(out) == 150
        assert max(sizes) <= 64

    @pytest.mark.asyncio
    async def test_order_survives_the_split(self):
        """Vectors are zipped positionally against entities in _write_document,
        so a reordering would attach each embedding to the wrong entity."""
        svc = LLMService(make_config(embedding_batch_size=10))

        async def record(texts):
            return [fake_embed(t) for t in texts]

        svc._embed_batch = record
        texts = [f"zeus {i}" for i in range(35)]
        out = await svc.get_embeddings(texts)
        assert out == [fake_embed(t) for t in texts]

    @pytest.mark.asyncio
    async def test_a_batch_within_the_limit_is_still_one_request(self):
        svc = LLMService(make_config(embedding_batch_size=64))
        calls = []

        async def record(texts):
            calls.append(len(texts))
            return [fake_embed(t) for t in texts]

        svc._embed_batch = record
        await svc.get_embeddings([f"e{i}" for i in range(9)])
        assert calls == [9], "batching must not add round trips below the cap"

    @pytest.mark.asyncio
    async def test_empty_input_makes_no_request(self):
        svc = LLMService(make_config())
        called = []
        svc._embed_batch = lambda texts: called.append(1)
        assert await svc.get_embeddings([]) == []
        assert not called


class TestDerivedEmbeddingsAreNotWorthThePassage:

    @pytest.mark.asyncio
    async def test_the_passage_survives_when_its_structure_cannot_be_embedded(self, rag_factory):
        rag = await rag_factory(extraction=RICH)

        async def boom(texts):
            raise EmbeddingError("embedding endpoint refused the batch")

        rag.llm.get_embeddings = boom          # derived only; get_embedding still works

        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))

        assert res["embeddings_degraded"] is True
        assert res["extraction_empty"] is True
        assert res["entities_extracted"] == 0
        assert res["relations_added"] == 0

        docs = await rag.store.client.get_vertices(
            "documents", realm=rag.config.realm, space="default")
        assert len(docs) == 1, "the passage must be kept"

        hits = await rag.store.search_similar_documents(fake_embed(TEXT), top_k=5, space="default")
        assert hits, "and must be retrievable by meaning"

    @pytest.mark.asyncio
    async def test_no_half_embedded_structure_reaches_the_graph(self, rag_factory):
        """An entity stored without a vector is invisible to the similarity
        search that finds entities, while looking complete in the graph."""
        rag = await rag_factory(extraction=RICH)

        async def boom(texts):
            raise EmbeddingError("endpoint refused the batch")

        rag.llm.get_embeddings = boom
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))

        ents = await rag.store.client.get_vertices(
            "entities", realm=rag.config.realm, space="default")
        assert ents == []
        assert await rag.store.get_all_relations(space="default") == []

    @pytest.mark.asyncio
    async def test_the_chunks_own_embedding_failing_still_fails_the_chunk(self, rag_factory):
        """The one case that must keep raising. A passage stored with no vector
        of its own is in the corpus and retrievable by nothing, which is the
        exact failure all of this exists to prevent."""
        rag = await rag_factory(extraction=RICH)

        async def boom(text):
            raise EmbeddingError("endpoint down")

        rag.llm.get_embedding = boom

        with pytest.raises(EmbeddingError):
            await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))

        docs = await rag.store.client.get_vertices(
            "documents", realm=rag.config.realm, space="default")
        assert docs == []

    @pytest.mark.asyncio
    async def test_the_two_flags_are_independent(self, rag_factory):
        """extraction_empty says no structure was written; embeddings_degraded
        says why. A caller needs to tell a corpus with little structure apart
        from an endpoint refusing work."""
        rag = await rag_factory(extraction=RICH)
        ok = await rag.index_document(TEXT, metadata=DocumentMetadata(document="a.txt"))
        assert (ok["extraction_empty"], ok["embeddings_degraded"]) == (False, False)

        rag2 = await rag_factory(extraction=ExtractionResult(entities=[], triples=[]))
        empty = await rag2.index_document(TEXT, metadata=DocumentMetadata(document="b.txt"))
        assert (empty["extraction_empty"], empty["embeddings_degraded"]) == (True, False)

    @pytest.mark.asyncio
    async def test_a_batch_of_chunks_loses_none_of_them(self, rag_factory):
        """The shape of the incident: every chunk degrades, none is lost."""
        rag = await rag_factory(extraction=RICH)

        async def boom(texts):
            raise EmbeddingError("endpoint refused the batch")

        rag.llm.get_embeddings = boom
        results = await rag.index_documents([
            (f"{TEXT} {i}", DocumentMetadata(document="myth.txt")) for i in range(4)
        ])
        assert len(results) == 4
        assert all(r["embeddings_degraded"] for r in results)
        docs = await rag.store.client.get_vertices(
            "documents", realm=rag.config.realm, space="default")
        assert len(docs) == 4


class TestTheWritePathNeverAbandonsAStoredPassage:
    """Seven operations run after add_document, one of them an LLM call.

    A failure there used to raise, which left the passage in the corpus and
    reported the chunk as skipped -- worse than losing it, because a caller
    comparing posted against indexed counts then fires on a mismatch that is
    actively wrong about what is stored. Structural failures are counted and
    reported instead, and a re-index restores what they cost.
    """

    @pytest.mark.asyncio
    async def test_a_failing_contradiction_check_keeps_chunk_and_relation(self, rag_factory):
        """The flakiest thing in the write path is the LLM call inside
        supersession. Losing it must cost neither the passage nor the relation
        it was enriching."""
        from post_graph_rag.errors import LLMError

        rag = await rag_factory(extraction=RICH, contradiction_detection=True)

        async def boom(*a, **k):
            raise LLMError("contradiction model is down")

        rag.extractor.detect_contradictions = boom

        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))

        assert res["document_id"]
        assert res["structure_errors"] >= 1
        assert res["relations_added"] == 1, "the relation was already stored"
        assert res["entities_extracted"] == 2

        rels = await rag.store.get_all_relations(space="default")
        assert [e.relation_type for e, _, _ in rels] == ["married_to"]

    @pytest.mark.asyncio
    async def test_a_failing_entity_write_costs_only_that_entity(self, rag_factory):
        rag = await rag_factory(extraction=RICH)
        real = rag.store.upsert_entity
        calls = {"n": 0}

        async def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("entity write failed")
            return await real(*a, **k)

        rag.store.upsert_entity = flaky
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))

        assert res["document_id"], "the passage survives"
        assert res["structure_errors"] == 1
        assert res["entities_extracted"] == 2, "counts report what was extracted"

        ents = await rag.store.client.get_vertices(
            "entities", realm=rag.config.realm, space="default")
        assert len(ents) == 1, "one entity was lost, not both and not the passage"

    @pytest.mark.asyncio
    async def test_structure_errors_is_zero_on_a_healthy_run(self, rag_factory):
        rag = await rag_factory(extraction=RICH)
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="myth.txt"))
        assert res["structure_errors"] == 0

    @pytest.mark.asyncio
    async def test_a_totally_broken_write_path_still_stores_every_passage(self, rag_factory):
        """The systematic case. Nothing raises, every passage lands, and
        structure_errors on every chunk is the signal -- which is why it has to
        be aggregated by the caller: no other count here would say the corpus
        is complete and useless."""
        rag = await rag_factory(extraction=RICH)

        async def boom(*a, **k):
            raise RuntimeError("entities table is unwritable")

        rag.store.upsert_entity = boom
        results = await rag.index_documents([
            (f"{TEXT} {i}", DocumentMetadata(document="myth.txt")) for i in range(3)
        ])

        assert len(results) == 3
        assert all(r["structure_errors"] > 0 for r in results)
        assert all(r["relations_added"] == 0 for r in results)
        docs = await rag.store.client.get_vertices(
            "documents", realm=rag.config.realm, space="default")
        assert len(docs) == 3
        ents = await rag.store.client.get_vertices(
            "entities", realm=rag.config.realm, space="default")
        assert ents == []
