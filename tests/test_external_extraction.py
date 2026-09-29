"""Structure that should never have been guessed at can be supplied directly.

Imports, call edges, ownership and deployment topology are recoverable exactly
from an AST, an LSP index or a build file. Asking a model to infer them is
strictly worse than reading them, and until now there was no way to hand them
in. Extraction earns its cost on the unwritten part -- rationale, causation,
what a decision was for -- which is what `merge` mode keeps.

What must not weaken: the gates. An externally supplied record goes through the
same validation as the LLM's unless a caller explicitly opts out, because the
gates are what make "extraction output is untrusted input" true of every source
rather than only of the model.
"""
import pytest
from conftest import fake_embed

from post_graph_rag import DocumentMetadata, QueryParam
from post_graph_rag.errors import ExtractionError
from post_graph_rag.extractor import Entity, ExtractionResult, GraphExtractor, Triple

TEXT = "The checkout service calls the ledger service."

LLM_SIDE = ExtractionResult(
    entities=[Entity(name="Checkout", type="Service", description="takes payment")],
    triples=[Triple(subject="Checkout", predicate="owned_by", object="Payments",
                    description="team ownership")],
)


def ast_edges(text, context=None):
    """What a static analyser would hand over: exact, and entity-free."""
    return [{"subject": "checkout_service", "predicate": "calls", "object": "ledger_service"}]


async def async_ast_edges(text, context=None):
    return ast_edges(text, context)


async def _relations(rag, space="default"):
    return await rag.store.get_all_relations(space=space)


class TestShapesAccepted:
    """A caller should not have to build a pydantic model to hand over triples."""

    @pytest.mark.asyncio
    async def test_a_bare_list_of_triple_dicts(self, rag_factory):
        rag = await rag_factory(extraction_fn=ast_edges)
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert res["triples_extracted"] == 1
        rels = await _relations(rag)
        assert [e.relation_type for e, _, _ in rels] == ["calls"]

    @pytest.mark.asyncio
    async def test_an_async_function(self, rag_factory):
        rag = await rag_factory(extraction_fn=async_ast_edges)
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert res["triples_extracted"] == 1

    @pytest.mark.asyncio
    async def test_a_mapping_of_entities_and_triples(self, rag_factory):
        def fn(text, context=None):
            return {
                "entities": [{"name": "Checkout", "type": "Service", "description": "takes payment"}],
                "triples": [{"subject": "Checkout", "predicate": "calls", "object": "Ledger"}],
            }
        rag = await rag_factory(extraction_fn=fn)
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert res["entities_extracted"] == 1
        assert res["triples_extracted"] == 1

    @pytest.mark.asyncio
    async def test_an_extraction_result(self, rag_factory):
        rag = await rag_factory(extraction_fn=lambda t, c=None: LLM_SIDE)
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert res["entities_extracted"] == 1

    def test_an_unusable_return_is_refused_not_guessed_at(self):
        with pytest.raises(ExtractionError):
            GraphExtractor._coerce_extraction(42)


class TestEndpointsBecomeStubs:

    @pytest.mark.asyncio
    async def test_triples_alone_still_connect_two_vertices(self, rag_factory):
        """Endpoints the caller did not describe take the same stub path the
        LLM's own unmatched endpoints take, so a triples-only source produces a
        traversable graph rather than dangling edges."""
        rag = await rag_factory(extraction_fn=ast_edges)
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))

        rels = await _relations(rag)
        assert len(rels) == 1
        edge, src, tgt = rels[0]
        assert src.payload["name"].lower() == "checkout_service"
        assert tgt.payload["name"].lower() == "ledger_service"


class TestModes:

    @pytest.mark.asyncio
    async def test_replace_does_not_call_the_llm_for_extraction(self, rag_factory):
        rag = await rag_factory(extraction=LLM_SIDE, extraction_fn=ast_edges,
                                extraction_fn_mode="replace")
        rag.llm.roles.clear()
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert "extraction" not in rag.llm.roles, "replace mode must skip the model"
        assert [e.relation_type for e, _, _ in await _relations(rag)] == ["calls"]

    @pytest.mark.asyncio
    async def test_merge_keeps_both_sources(self, rag_factory):
        """The composition case: a deterministic code graph and the model's
        reading of the prose around it, in one graph."""
        rag = await rag_factory(extraction=LLM_SIDE, extraction_fn=ast_edges,
                                extraction_fn_mode="merge")
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        kinds = sorted(e.relation_type for e, _, _ in await _relations(rag))
        assert kinds == ["calls", "owned_by"], "both sources must survive the union"

    def test_an_unknown_mode_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            GraphExtractor(llm_service=None, extraction_fn=ast_edges,
                           extraction_fn_mode="both")


class TestGates:

    @pytest.mark.asyncio
    async def test_external_records_are_gated_by_default(self, rag_factory):
        """A pronominal endpoint is dropped from an external source exactly as
        it is from the model's."""
        def fn(text, context=None):
            return [
                {"subject": "checkout_service", "predicate": "calls", "object": "ledger_service"},
                {"subject": "he", "predicate": "calls", "object": "it"},
            ]
        rag = await rag_factory(extraction_fn=fn)
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        rels = await _relations(rag)
        assert len(rels) == 1, "the pronominal triple must not reach the graph"

    @pytest.mark.asyncio
    async def test_the_gates_can_be_turned_off_for_an_exact_source(self, rag_factory):
        """An AST's `calls` should not be snapped onto a prose vocabulary."""
        def fn(text, context=None):
            return [{"subject": "checkout_service", "predicate": "calls", "object": "ledger_service"}]
        rag = await rag_factory(extraction_fn=fn, validate_external_extraction=False,
                                predicate_vocabulary=["worked_with", "located_in"])
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert [e.relation_type for e, _, _ in await _relations(rag)] == ["calls"]


class TestFailuresAndProvenance:

    @pytest.mark.asyncio
    async def test_a_broken_extractor_fails_the_chunk(self, rag_factory):
        """A broken external extractor is the same event as a dead model: the
        chunk could not be read. Swallowing it would let an import that stopped
        resolving read as a corpus with no call edges in it."""
        def fn(text, context=None):
            raise RuntimeError("the language server died")
        rag = await rag_factory(extraction_fn=fn)
        with pytest.raises(RuntimeError):
            await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))

    @pytest.mark.asyncio
    async def test_returning_nothing_keeps_the_passage(self, rag_factory):
        """Consistent with 1.14.2: nothing extractable is a fact about the
        passage, not a failed run."""
        rag = await rag_factory(extraction_fn=lambda t, c=None: [])
        res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
        assert res["extraction_empty"] is True
        assert res["triples_extracted"] == 0
        hits = await rag.store.search_similar_documents(fake_embed(TEXT), top_k=5, space="default")
        assert hits, "the passage is still retrievable"

    @pytest.mark.asyncio
    async def test_external_relations_carry_provenance(self, rag_factory):
        """Without a source chunk an external relation could never be withdrawn
        by a deletion -- the stranded-row problem, reintroduced at the door."""
        rag = await rag_factory(extraction_fn=ast_edges)
        await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))

        edge = (await _relations(rag))[0][0]
        ref = rag.store.client._get_table_ref("relations", rag.config.realm)
        rows = await rag.store.client._fetch(
            f"SELECT payload FROM {ref} WHERE realm = $1 AND id = $2",
            rag.config.realm, int(edge.id))
        import json as _json
        pl = rows[0]["payload"]
        pl = pl if isinstance(pl, dict) else _json.loads(pl)
        assert pl["sources"], "an external relation must record the chunk that asserted it"

        await rag.store.delete_document_chunks("svc.py", space="default")
        assert await _relations(rag) == [], "and must therefore be withdrawable"


@pytest.mark.asyncio
async def test_nothing_changes_when_no_function_is_configured(rag_factory):
    """The default path is untouched."""
    rag = await rag_factory(extraction=LLM_SIDE)
    res = await rag.index_document(TEXT, metadata=DocumentMetadata(document="svc.py"))
    assert res["entities_extracted"] == 1
    assert [e.relation_type for e, _, _ in await _relations(rag)] == ["owned_by"]
    out = await rag.query("who owns checkout?", param=QueryParam(mode="mix"))
    assert out["answer"]
