"""Integration and unit tests for post-graph-rag v1.20.0 core capabilities.

Exercises belief epochs (A -> B -> A reverts and polarity flips), functional
predicate supersession and bare-quantity object validation, conflicting-edge
immutability, realm embedding-model fingerprinting, concurrency-safe direct
extractions, bitemporal point-in-time retrieval (as_believed_at and as_of),
and inbound/bidirectional neighborhood traversal against live PostgreSQL.
"""
import asyncio
import json

import pytest
from conftest import VOCAB_DIM, FakeLLM, make_config

import post_graph_rag
from post_graph_rag import (
    DocumentMetadata,
    Entity,
    ExtractionResult,
    GraphExtractor,
    GraphRAG,
    QueryParam,
    RAGConfig,
    RAGGraphStore,
    SchemaError,
    Triple,
    normalise_predicate,
)
from post_graph_rag.errors import ExtractionError


def _single_triple(
    subject: str,
    predicate: str,
    obj: str,
    *,
    negated: bool = False,
    valid_from: str | None = None,
    valid_to: str | None = None,
    subj_type: str = "Technology",
    obj_type: str = "Technology",
    include_obj_entity: bool = True,
) -> ExtractionResult:
    entities = [Entity(name=subject, type=subj_type, description=f"{subject} entity")]
    if include_obj_entity:
        entities.append(Entity(name=obj, type=obj_type, description=f"{obj} entity"))
    return ExtractionResult(
        entities=entities,
        triples=[
            Triple(
                subject=subject,
                predicate=predicate,
                object=obj,
                description=f"{subject} {predicate} {obj}",
                negated=negated,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        ],
    )


# ------------------------------------------------------- version & config


class TestVersionAndConfig:
    def test_package_version_is_1_20_0(self):
        assert post_graph_rag.__version__ == "1.20.0"
        assert normalise_predicate("has_role") == "role"

    def test_functional_predicates_defaults_and_env(self, monkeypatch):
        monkeypatch.delenv("RAG_FUNCTIONAL_PREDICATES", raising=False)
        cfg_default = RAGConfig()
        assert cfg_default.functional_predicates == set()

        monkeypatch.setenv("RAG_FUNCTIONAL_PREDICATES", "uses_auth, port_is , ")
        cfg_env = RAGConfig()
        assert cfg_env.functional_predicates == {"uses_auth", "port_is"}

    def test_enforce_embedding_model_match_defaults_and_env(self, monkeypatch):
        monkeypatch.delenv("RAG_ENFORCE_EMBEDDING_MODEL", raising=False)
        assert RAGConfig().enforce_embedding_model_match is True

        monkeypatch.setenv("RAG_ENFORCE_EMBEDDING_MODEL", "0")
        assert RAGConfig().enforce_embedding_model_match is False

    def test_query_param_direction_default(self):
        p = QueryParam()
        assert p.direction == "out"

    @pytest.mark.asyncio
    async def test_query_data_rejects_invalid_direction(self, rag_factory):
        rag = await rag_factory()
        with pytest.raises(ValueError, match="direction"):
            await rag.query_data("anything", QueryParam(direction="sideways"))


# -------------------------------------- extractor functional bare quantities


class TestFunctionalBareQuantityValidation:
    def test_functional_predicate_objects_survive_bare_quantity_gate(self):
        cfg = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
        extractor = GraphExtractor(
            FakeLLM(cfg),
            gleaning_passes=0,
            functional_predicates={"port_is", "has_api_version"},
        )
        assert extractor.functional_predicates == {"port_is", "api_version"}

        raw = ExtractionResult(
            entities=[
                Entity(name="ApiServer", type="Technology", description="HTTP backend"),
                Entity(name="8080", type="Concept", description="listen port"),
                Entity(name="PublicApi", type="Technology", description="REST surface"),
                Entity(name="$18.4 billion", type="Concept", description="prose revenue"),
            ],
            triples=[
                Triple(subject="ApiServer", predicate="port_is", object="8080"),
                Triple(subject="PublicApi", predicate="api_version", object="2"),
                Triple(subject="ApiServer", predicate="reported_revenue", object="$18.4 billion"),
                Triple(subject="ApiServer", predicate="margin_was", object="40%"),
                Triple(subject="8080", predicate="port_is", object="ApiServer"),
            ],
        )
        cleaned = extractor._validate(raw)
        kept = {(t.subject, t.predicate, t.object) for t in cleaned.triples}
        assert kept == {
            ("ApiServer", "port_is", "8080"),
            ("PublicApi", "api_version", "2"),
        }

    def test_sec_prose_bare_figures_still_rejected_without_functional_match(self):
        cfg = RAGConfig(api_base="http://localhost:9/v1", api_key="k", embedding_dim=16)
        extractor = GraphExtractor(
            FakeLLM(cfg),
            gleaning_passes=0,
            functional_predicates={"port_is"},
        )
        raw = ExtractionResult(
            entities=[Entity(name="Calder Corp", type="Organization", description="issuer")],
            triples=[
                Triple(subject="Calder Corp", predicate="reported_revenue", object="$412 million")
            ],
        )
        cleaned = extractor._validate(raw)
        assert [e.name for e in cleaned.entities] == ["Calder Corp"]
        assert cleaned.triples == []

        only_figure = ExtractionResult(
            entities=[Entity(name="$412 million", type="Concept", description="figure")],
            triples=[
                Triple(subject="Calder Corp", predicate="reported_revenue", object="$412 million")
            ],
        )
        with pytest.raises(ExtractionError):
            extractor._validate(only_figure)

    @pytest.mark.asyncio
    async def test_set_functional_predicates_and_config_sync_end_to_end(self, rag_factory):
        rag = await rag_factory()
        rag.set_functional_predicates({"port_is"})
        assert "port_is" in rag.extractor.functional_predicates

        res = await rag.index_document(
            "ApiServer listens on 8080.",
            metadata=DocumentMetadata(document="port-1.md"),
            extraction=_single_triple("ApiServer", "port_is", "8080"),
        )
        assert res["relations_added"] == 1
        assert len(res["added_edge_ids"]) == 1

        # Direct mutation on config also syncs before validation and supersession.
        rag.config.functional_predicates.add("api_version")
        res2 = await rag.index_document(
            "ApiServer moved to port 9090 and API v2.",
            metadata=DocumentMetadata(document="port-2.md"),
            extraction=ExtractionResult(
                entities=[Entity(name="ApiServer", type="Technology", description="HTTP server")],
                triples=[
                    Triple(subject="ApiServer", predicate="port_is", object="9090"),
                    Triple(subject="ApiServer", predicate="api_version", object="2"),
                ],
            ),
        )
        assert res2["relations_added"] == 2
        assert res2["relations_superseded"] == 1
        assert res2["superseded_details"][0]["old_object"] == "8080"
        assert res2["superseded_details"][0]["reason"] == "functional_predicate"


# -------------------------------------- realm embedding_model fingerprint


class TestEmbeddingModelFingerprintGuard:
    @pytest.mark.asyncio
    async def test_same_dim_different_embedding_model_raises_schema_error(self, rag_factory):
        rag = await rag_factory(embedding_model="BAAI/bge-small-en-v1.5")
        realm = rag.config.realm

        # Re-initializing with the exact same model is idempotent and keeps one metadata row.
        await rag.store.initialize_schema()
        rows = await rag.store.client.find_vertices(
            "realm_metadata",
            realm=realm,
            space="_system",
            filters={"key": "embedding_fingerprint"},
        )
        assert len(rows) == 1
        assert rows[0].payload["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert rows[0].payload["embedding_dim"] == VOCAB_DIM

        # Same vector dimension, different model -> SchemaError naming both models and realm.
        mismatched = RAGGraphStore(
            make_config(
                realm=realm,
                embedding_dim=VOCAB_DIM,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        await mismatched.connect()
        try:
            with pytest.raises(SchemaError) as exc_info:
                await mismatched.initialize_schema()
            msg = str(exc_info.value)
            assert realm in msg
            assert "BAAI/bge-small-en-v1.5" in msg
            assert "sentence-transformers/all-MiniLM-L6-v2" in msg
        finally:
            await mismatched.close()

        # Opting out explicitly allows opening the realm without error.
        override_store = RAGGraphStore(
            make_config(
                realm=realm,
                embedding_dim=VOCAB_DIM,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                enforce_embedding_model_match=False,
            )
        )
        await override_store.connect()
        try:
            await override_store.initialize_schema()
        finally:
            await override_store.close()


# --------------------------- belief epochs: reverts & polarity flips


class TestBeliefEpochsRevertsAndPolarityFlips:
    @pytest.mark.asyncio
    async def test_a_to_b_to_a_revert_creates_new_epoch_without_deadlock(self, rag_factory):
        rag = await rag_factory(functional_predicates={"uses_auth"})

        r1 = await rag.index_document(
            "ADR 0001: AuthService uses Clerk.",
            metadata=DocumentMetadata(document="adr-0001.md"),
            extraction=_single_triple("AuthService", "uses_auth", "Clerk"),
        )
        edge_1 = r1["added_edge_ids"][0]
        assert r1["superseded_ids"] == []

        r2 = await rag.index_document(
            "ADR 0002: AuthService migrated to BetterAuth.",
            metadata=DocumentMetadata(document="adr-0002.md"),
            extraction=_single_triple("AuthService", "uses_auth", "BetterAuth"),
        )
        edge_2 = r2["added_edge_ids"][0]
        assert r2["superseded_ids"] == [edge_1]
        assert r2["superseded_details"][0]["old_object"] == "Clerk"
        assert r2["superseded_details"][0]["reason"] == "functional_predicate"

        e1_after_r2 = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=edge_1
        )
        t_expired_1 = e1_after_r2.payload["t_expired"]
        assert e1_after_r2.payload["superseded_by"] == edge_2
        assert t_expired_1 is not None

        # Revert back to Clerk: must open a 3rd edge epoch, close BetterAuth,
        # and leave Epoch 1's superseded_by and t_expired untouched.
        r3 = await rag.index_document(
            "ADR 0003: AuthService reverted from BetterAuth back to Clerk.",
            metadata=DocumentMetadata(document="adr-0003.md"),
            extraction=_single_triple("AuthService", "uses_auth", "Clerk"),
        )
        edge_3 = r3["added_edge_ids"][0]
        assert edge_3 not in {edge_1, edge_2}
        assert r3["superseded_ids"] == [edge_2]
        assert r3["superseded_details"][0]["old_object"] == "BetterAuth"

        e1_after_r3 = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=edge_1
        )
        e2_after_r3 = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=edge_2
        )
        e3_after_r3 = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=edge_3
        )
        assert e1_after_r3.payload["superseded_by"] == edge_2
        assert e1_after_r3.payload["t_expired"] == t_expired_1
        assert e2_after_r3.payload["superseded_by"] == edge_3
        assert e3_after_r3.payload.get("superseded_by") is None

        # Present-day recall returns strictly the active Clerk epoch.
        current = await rag.query_data(
            "Which auth provider does AuthService use?",
            QueryParam(ll_keywords=["AuthService"], hl_keywords=["auth"]),
        )
        rels = current["data"]["relationships"]
        assert [(r["tgt_id"], r["edge_id"], r["superseded_by"]) for r in rels] == [
            ("Clerk", edge_3, None)
        ]

        # Full history includes all 3 distinct epochs (2 superseded, 1 active).
        history = await rag.query_data(
            "AuthService auth history",
            QueryParam(
                ll_keywords=["AuthService"],
                hl_keywords=["auth"],
                include_superseded=True,
                direction="both",
            ),
        )
        hist_ids = {r["edge_id"] for r in history["data"]["relationships"]}
        assert hist_ids == {edge_1, edge_2, edge_3}

    @pytest.mark.asyncio
    async def test_polarity_flip_supersedes_without_erasing_negation(self, rag_factory):
        rag = await rag_factory()

        r1 = await rag.index_document(
            "BillingService uses Redis cache.",
            metadata=DocumentMetadata(document="cache-1.md"),
            extraction=_single_triple("BillingService", "uses_cache", "Redis", negated=False),
        )
        pos_id = r1["added_edge_ids"][0]

        r2 = await rag.index_document(
            "BillingService no longer uses Redis cache.",
            metadata=DocumentMetadata(document="cache-2.md"),
            extraction=_single_triple("BillingService", "uses_cache", "Redis", negated=True),
        )
        neg_id = r2["added_edge_ids"][0]
        assert neg_id != pos_id
        assert r2["superseded_ids"] == [pos_id]
        assert r2["superseded_details"] == [
            {
                "id": pos_id,
                "subject": "BillingService",
                "predicate": "uses_cache",
                "old_object": "Redis",
                "old_negated": False,
                "closed_by_edge_id": neg_id,
                "reason": "polarity_flip",
            }
        ]

        now = await rag.query_data(
            "Does BillingService use Redis?",
            QueryParam(ll_keywords=["BillingService"], hl_keywords=["cache"]),
        )
        active = now["data"]["relationships"]
        assert len(active) == 1
        assert active[0]["edge_id"] == neg_id
        assert active[0]["negated"] is True

        # Re-enabling flips polarity back and closes the negated epoch.
        r3 = await rag.index_document(
            "BillingService re-enabled Redis cache.",
            metadata=DocumentMetadata(document="cache-3.md"),
            extraction=_single_triple("BillingService", "uses_cache", "Redis", negated=False),
        )
        pos2_id = r3["added_edge_ids"][0]
        assert pos2_id not in {pos_id, neg_id}
        assert r3["superseded_ids"] == [neg_id]
        assert r3["superseded_details"][0]["old_negated"] is True
        assert r3["superseded_details"][0]["reason"] == "polarity_flip"


# ---------------- conflicting immutability & tense normalization


class TestSupersessionNormalizationAndImmutability:
    @pytest.mark.asyncio
    async def test_supersede_conflicting_never_overwrites_already_retracted_edge(self, rag_factory):
        rag = await rag_factory(
            exclusive_predicate_groups=[{"approved", "deprecated", "rejected"}]
        )
        r1 = await rag.index_document(
            "RFC-12 approved.",
            metadata=DocumentMetadata(document="rfc-1.md"),
            extraction=_single_triple("PlatformTeam", "approved", "RFC-12"),
        )
        e1 = r1["added_edge_ids"][0]

        r2 = await rag.index_document(
            "RFC-12 deprecated.",
            metadata=DocumentMetadata(document="rfc-2.md"),
            extraction=_single_triple("PlatformTeam", "deprecated", "RFC-12"),
        )
        e2 = r2["added_edge_ids"][0]
        assert r2["superseded_ids"] == [e1]
        assert r2["superseded_details"][0]["reason"] == "exclusive_predicate_group"

        e1_mid = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=e1
        )
        e1_expired = e1_mid.payload["t_expired"]
        assert e1_mid.payload["superseded_by"] == e2

        r3 = await rag.index_document(
            "RFC-12 rejected.",
            metadata=DocumentMetadata(document="rfc-3.md"),
            extraction=_single_triple("PlatformTeam", "rejected", "RFC-12"),
        )
        e3 = r3["added_edge_ids"][0]
        assert r3["superseded_ids"] == [e2]

        e1_final = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=e1
        )
        assert e1_final.payload["superseded_by"] == e2
        assert e1_final.payload["t_expired"] == e1_expired

        e2_final = await rag.store.client.get_edge(
            "relations", realm=rag.config.realm, edge_id=e2
        )
        assert e2_final.payload["superseded_by"] == e3

    @pytest.mark.asyncio
    async def test_tense_prefix_normalization_in_functional_supersession(self, rag_factory):
        rag = await rag_factory(functional_predicates={"has_role"})
        # Store-level raw predicates differing by TENSE_PREFIXES ("has_role" vs "role")
        # still match via normalise_predicate inside supersede_functional.
        rag.config.validate_external_extraction = False

        r1 = await rag.index_document(
            "Dana has role Staff Engineer.",
            metadata=DocumentMetadata(document="role-1.md"),
            extraction=_single_triple(
                "Dana", "has_role", "Staff Engineer", subj_type="Person", obj_type="Concept"
            ),
        )
        e1 = r1["added_edge_ids"][0]

        r2 = await rag.index_document(
            "Dana role Principal Engineer.",
            metadata=DocumentMetadata(document="role-2.md"),
            extraction=_single_triple(
                "Dana", "role", "Principal Engineer", subj_type="Person", obj_type="Concept"
            ),
        )
        assert r2["superseded_ids"] == [e1]
        assert r2["superseded_details"][0]["old_object"] == "Staff Engineer"


# ------- concurrent direct extraction, retraction & active inspection


class TestConcurrentDirectExtractionAndHelpers:
    @pytest.mark.asyncio
    async def test_concurrent_index_document_with_extraction_uses_zero_chat_llm(self, rag_factory):
        rag = await rag_factory()
        rag.llm.chat_calls.clear()

        tasks = [
            rag.index_document(
                f"Service-{i} depends on Library-{i}.",
                metadata=DocumentMetadata(document=f"svc-{i}.md"),
                extraction=_single_triple(f"Service-{i}", "depends_on", f"Library-{i}"),
            )
            for i in range(8)
        ]
        outcomes = await asyncio.gather(*tasks)
        assert len(outcomes) == 8
        assert rag.llm.chat_calls == []
        for out in outcomes:
            assert out["relations_added"] == 1
            assert len(out["added_edge_ids"]) == 1

    @pytest.mark.asyncio
    async def test_active_relations_for_entities_and_manual_retract(self, rag_factory):
        rag = await rag_factory()
        r1 = await rag.index_document(
            "AuthService uses custom predicate auth_lib Clerk.",
            metadata=DocumentMetadata(document="m1.md"),
            extraction=ExtractionResult(
                entities=[
                    Entity(
                        name="AuthService",
                        type="Technology",
                        description="auth",
                        aliases=["AuthSvc"],
                    ),
                    Entity(name="Clerk", type="Technology", description="vendor"),
                ],
                triples=[Triple(subject="AuthService", predicate="auth_lib", object="Clerk")],
            ),
        )
        clerk_edge_id = r1["added_edge_ids"][0]

        r2 = await rag.index_document(
            "AuthService migrated auth_engine to BetterAuth.",
            metadata=DocumentMetadata(document="m2.md"),
            extraction=_single_triple("AuthService", "auth_engine", "BetterAuth"),
        )
        better_edge_id = r2["added_edge_ids"][0]

        # Lookup by alias ("AuthSvc") resolves canonical name and excludes newly added edge.
        active_map = await rag.active_relations_for_entities(
            ["AuthSvc"], exclude_edge_ids={better_edge_id}, limit_per_entity=5
        )
        assert "AuthService" in active_map
        assert [x["id"] for x in active_map["AuthService"]] == [clerk_edge_id]
        assert active_map["AuthService"][0]["object"] == "Clerk"

        # Host-driven manual retraction closes the older synonymous edge.
        retracted = await rag.retract_relations([clerk_edge_id], better_edge_id)
        assert retracted == [clerk_edge_id]

        after = await rag.active_relations_for_entities(["AuthService"])
        assert [x["id"] for x in after["AuthService"]] == [better_edge_id]

        # All-documents stats works when doc_keys is omitted.
        all_stats = await rag.documents_stats()
        assert {"m1.md", "m2.md"} <= set(all_stats.keys())

    def test_llm_service_keyword_alias_accepted_on_graph_rag_init(self):
        cfg = make_config()
        fake = FakeLLM(cfg)
        rag = GraphRAG(cfg, llm_service=fake)
        assert rag.llm is fake


# ----------------- bitemporal point-in-time retrieval & direction


class TestBitemporalRecallAndDirection:
    @pytest.mark.asyncio
    async def test_as_believed_at_recovers_superseded_belief_state(self, rag_factory):
        rag = await rag_factory(functional_predicates={"cfo"})

        await rag.index_document(
            "Calder Corp appointed Voss as CFO.",
            metadata=DocumentMetadata(document="filing-1.md"),
            extraction=_single_triple(
                "Calder Corp", "cfo", "Voss", subj_type="Organization", obj_type="Person"
            ),
        )
        watermark = await rag.watermark()
        await asyncio.sleep(0.02)

        await rag.index_document(
            "Calder Corp restated leadership: Nair replaced Voss as CFO.",
            metadata=DocumentMetadata(document="filing-2.md"),
            extraction=_single_triple(
                "Calder Corp", "cfo", "Nair", subj_type="Organization", obj_type="Person"
            ),
        )

        # Present-day query sees Nair only.
        present = await rag.query_data(
            "Who is CFO of Calder Corp?",
            QueryParam(ll_keywords=["Calder Corp"], hl_keywords=["cfo"]),
        )
        assert [r["tgt_id"] for r in present["data"]["relationships"]] == ["Nair"]

        # Historical belief-time query at watermark sees Voss across both 1-hop and 2-hop walks.
        for hops in (1, 2):
            believed = await rag.query_data(
                "Who is CFO of Calder Corp?",
                QueryParam(
                    as_believed_at=watermark,
                    max_hops=hops,
                    ll_keywords=["Calder Corp"],
                    hl_keywords=["cfo"],
                ),
            )
            assert [r["tgt_id"] for r in believed["data"]["relationships"]] == ["Voss"]

    @pytest.mark.asyncio
    async def test_as_of_retains_dated_superseded_edges_and_drops_undated_superseded(
        self, rag_factory
    ):
        rag = await rag_factory(functional_predicates={"cfo", "uses_erp"})

        await rag.index_document(
            "Voss served as CFO of Calder Corp from 2019 to 2020; LegacyERP was in use.",
            metadata=DocumentMetadata(document="2019.md"),
            extraction=ExtractionResult(
                entities=[
                    Entity(name="Calder Corp", type="Organization", description="firm"),
                    Entity(name="Voss", type="Person", description="CFO"),
                    Entity(name="LegacyERP", type="Technology", description="old ERP"),
                ],
                triples=[
                    Triple(
                        subject="Calder Corp",
                        predicate="cfo",
                        object="Voss",
                        valid_from="2019-01-01",
                        valid_to="2020-12-31",
                    ),
                    Triple(subject="Calder Corp", predicate="uses_erp", object="LegacyERP"),
                ],
            ),
        )
        await rag.index_document(
            "Chandran served as CFO of Calder Corp from 2021 to 2022; migrated to ModernERP.",
            metadata=DocumentMetadata(document="2021.md"),
            extraction=ExtractionResult(
                entities=[
                    Entity(name="Calder Corp", type="Organization", description="firm"),
                    Entity(name="Chandran", type="Person", description="CFO"),
                    Entity(name="ModernERP", type="Technology", description="new ERP"),
                ],
                triples=[
                    Triple(
                        subject="Calder Corp",
                        predicate="cfo",
                        object="Chandran",
                        valid_from="2021-01-01",
                        valid_to="2022-12-31",
                    ),
                    Triple(subject="Calder Corp", predicate="uses_erp", object="ModernERP"),
                ],
            ),
        )

        res_2019 = await rag.query_data(
            "Calder Corp leadership and systems in 2019",
            QueryParam(
                as_of="2019-06-01",
                ll_keywords=["Calder Corp"],
                hl_keywords=["cfo", "erp"],
            ),
        )
        targets_2019 = {
            (r["relation_type"], r["tgt_id"]) for r in res_2019["data"]["relationships"]
        }
        assert ("cfo", "Voss") in targets_2019
        assert ("cfo", "Chandran") not in targets_2019
        assert ("uses_erp", "ModernERP") in targets_2019
        assert ("uses_erp", "LegacyERP") not in targets_2019

    @pytest.mark.asyncio
    async def test_inbound_and_both_direction_preserve_true_subject_object_orientation(
        self, rag_factory
    ):
        # Disable relation-embedding & lexical channels so ONLY entity traversal can find edges.
        rag = await rag_factory(
            embed_relations=False,
            lexical_search=False,
            merge_strategy="quota",
            relation_seed_quota=0.0,
        )
        # Use vocab-aligned tokens ("zeus", "hera", "poseidon") so vector search
        # seeds strictly on Poseidon (the object vertex) when queried for "Poseidon".
        await rag.index_document(
            "Zeus allied with Poseidon.",
            metadata=DocumentMetadata(document="d1.md"),
            extraction=_single_triple("Zeus", "allied_with", "Poseidon", subj_type="Person", obj_type="Person"),
        )
        await rag.index_document(
            "Hera consulted Poseidon.",
            metadata=DocumentMetadata(document="d2.md"),
            extraction=_single_triple("Hera", "consulted", "Poseidon", subj_type="Person", obj_type="Person"),
        )

        poseidon = await rag.store.find_entity("Poseidon")
        assert poseidon is not None

        # 1. Store-level 1-hop & 2-hop checks:
        assert await rag.store.get_neighborhood(poseidon.id, max_hops=1, direction="out") == []
        in_1 = await rag.store.get_neighborhood(poseidon.id, max_hops=1, direction="in")
        assert {(s.payload["name"], e.relation_type, t.payload["name"]) for e, s, t, _ in in_1} == {
            ("Zeus", "allied_with", "Poseidon"),
            ("Hera", "consulted", "Poseidon"),
        }
        both_2 = await rag.store.get_neighborhood(poseidon.id, max_hops=2, direction="both")
        assert {
            (s.payload["name"], e.relation_type, t.payload["name"]) for e, s, t, _ in both_2
        } == {
            ("Zeus", "allied_with", "Poseidon"),
            ("Hera", "consulted", "Poseidon"),
        }

        # 2. Engine-level query_data seeded purely on Poseidon (top_k=1):
        out_only = await rag.query_data(
            "Poseidon sea",
            QueryParam(top_k=1, max_hops=1, direction="out", ll_keywords=["Poseidon"], hl_keywords=["sea"]),
        )
        assert out_only["data"]["relationships"] == []

        both_q = await rag.query_data(
            "Poseidon sea",
            QueryParam(top_k=1, max_hops=1, direction="both", ll_keywords=["Poseidon"], hl_keywords=["sea"]),
        )
        found_pairs = {
            (r["src_id"], r["relation_type"], r["tgt_id"])
            for r in both_q["data"]["relationships"]
        }
        assert found_pairs == {
            ("Zeus", "allied_with", "Poseidon"),
            ("Hera", "consulted", "Poseidon"),
        }
        # Confirm JSON serializability of full query_data output.
        assert json.dumps(both_q)
