"""One model for every call is a compromise no role wants.

Extraction and synthesis have different optima, and the difference is measured:
over one fixed graph four answering models separate across a 23-point band,
while on the extraction side the models disagree about what a good graph even
is -- one builds the richest, another the most queryable. The volumes differ by
an order of magnitude too, since extraction runs once per chunk and the
query-time roles run on every question for ever.

These tests hold the contract that makes that expressible: a role routes to its
own model when one is configured, falls through to `model` when it is not, and
every LLM call site declares which role it is.
"""
import pytest
from conftest import make_config

from post_graph_rag import DocumentMetadata, QueryParam, RAGConfig
from post_graph_rag.extractor import Entity, ExtractionResult, Triple

ZEUS_DOC = "Zeus is king of the Olympian gods, married to Hera."
ZEUS_EXTRACTION = ExtractionResult(
    entities=[
        Entity(name="Zeus", type="Person", description="king of the Olympian gods"),
        Entity(name="Hera", type="Person", description="wife of Zeus"),
    ],
    triples=[Triple(subject="Zeus", predicate="married_to", object="Hera",
                    description="spouse")],
)


class TestResolution:

    def test_every_role_falls_through_to_model_by_default(self):
        """The whole point of the default: an existing single-model config must
        behave exactly as it did before roles existed."""
        c = make_config(model="only-model")
        for role in c.MODEL_ROLES:
            assert c.model_for(role) == "only-model"
        assert c.model_for() == "only-model"
        assert c.model_for(None) == "only-model"

    def test_an_override_routes_only_its_own_role(self):
        c = make_config(model="base", extraction_model="reasoner")
        assert c.model_for("extraction") == "reasoner"
        assert c.model_for("synthesis") == "base"
        assert c.model_for("auxiliary") == "base"
        assert c.model_for("community") == "base"

    def test_roles_are_independent(self):
        c = make_config(model="base", extraction_model="a", synthesis_model="b",
                        auxiliary_model="c", community_model="d")
        assert [c.model_for(r) for r in ("extraction", "synthesis", "auxiliary", "community")] \
            == ["a", "b", "c", "d"]

    def test_an_unknown_role_raises_rather_than_falling_back(self):
        """A typo at a call site would otherwise route that role to the default
        model for ever, and look like it was working."""
        c = make_config(model="base")
        with pytest.raises(ValueError) as exc:
            c.model_for("sythesis")
        assert "sythesis" in str(exc.value)
        assert "synthesis" in str(exc.value)      # names the valid roles

    def test_empty_env_override_is_treated_as_absent(self):
        """RAG_EXTRACTION_MODEL="" must mean "unset", not a model named ""."""
        c = make_config(model="base", extraction_model=None)
        assert c.model_for("extraction") == "base"

    def test_model_roles_is_not_a_dataclass_field(self):
        """It is a lookup table, not configuration; if it became a field it
        would appear in every config repr and be overridable by accident."""
        import dataclasses
        names = {f.name for f in dataclasses.fields(RAGConfig)}
        assert "MODEL_ROLES" not in names
        assert {"extraction_model", "synthesis_model",
                "auxiliary_model", "community_model"} <= names


class TestCandidateOrdering:

    def test_the_roles_model_leads_the_candidate_list(self):
        from post_graph_rag.llm import LLMService
        c = make_config(model="base", extraction_model="reasoner",
                        fallback_models=["fb1", "fb2"])
        svc = LLMService(c)
        assert svc._model_candidates("extraction") == ["reasoner", "fb1", "fb2"]
        assert svc._model_candidates("synthesis") == ["base", "fb1", "fb2"]
        assert svc._model_candidates() == ["base", "fb1", "fb2"]

    def test_a_role_model_that_repeats_a_fallback_is_not_duplicated(self):
        from post_graph_rag.llm import LLMService
        c = make_config(model="base", synthesis_model="fb1", fallback_models=["fb1", "fb2"])
        assert LLMService(c)._model_candidates("synthesis") == ["fb1", "fb2"]


class TestCallSitesDeclareTheirRole:

    @pytest.mark.asyncio
    async def test_indexing_uses_the_extraction_model(self, rag_factory):
        rag = await rag_factory(extraction=ZEUS_EXTRACTION,
                                model="base", extraction_model="reasoner")
        await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))

        assert "extraction" in rag.llm.roles
        assert "reasoner" in rag.llm.models_used
        assert "synthesis" not in rag.llm.roles, "indexing must not invoke the reader"

    @pytest.mark.asyncio
    async def test_querying_uses_synthesis_and_auxiliary_not_extraction(self, rag_factory):
        rag = await rag_factory(extraction=ZEUS_EXTRACTION, answer="an answer",
                                model="base", extraction_model="reasoner",
                                synthesis_model="instruct", auxiliary_model="small")
        await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))
        rag.llm.roles.clear()
        rag.llm.models_used.clear()

        await rag.query("Who is Zeus married to?", param=QueryParam(mode="mix"))

        assert "synthesis" in rag.llm.roles
        assert "instruct" in rag.llm.models_used
        assert "reasoner" not in rag.llm.models_used, \
            "the query path must never reach the extraction model"

    @pytest.mark.asyncio
    async def test_community_reports_use_the_community_model(self, rag_factory):
        rag = await rag_factory(extraction=ZEUS_EXTRACTION,
                                model="base", community_model="summariser")
        await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))
        rag.llm.roles.clear()
        rag.llm.models_used.clear()

        await rag.build_communities()

        if "community" in rag.llm.roles:          # only if a cluster was large enough
            assert "summariser" in rag.llm.models_used

    @pytest.mark.asyncio
    async def test_a_single_model_config_still_routes_everything_to_it(self, rag_factory):
        """Backwards compatibility, end to end: no overrides means every role
        resolves to `model`, exactly as before this existed."""
        rag = await rag_factory(extraction=ZEUS_EXTRACTION, answer="an answer",
                                model="only-model")
        await rag.index_document(ZEUS_DOC, metadata=DocumentMetadata(document="zeus.pdf"))
        await rag.query("Who is Zeus married to?", param=QueryParam(mode="mix"))

        assert set(rag.llm.models_used) == {"only-model"}
        assert rag.llm.roles, "call sites must declare a role even with one model"
