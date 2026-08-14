"""Tests for post_graph_rag.config — pure unit tests."""



from post_graph_rag.config import RAGConfig


class TestRAGConfigDefaults:
    def test_default_model(self):
        c = RAGConfig()
        assert isinstance(c.model, str)
        assert len(c.model) > 0

    def test_default_embedding_dim(self):
        c = RAGConfig()
        assert c.embedding_dim == 1536

    def test_default_max_hops(self):
        c = RAGConfig()
        assert c.max_hops == 2

    def test_default_space(self):
        c = RAGConfig()
        assert c.space == "default"

    def test_default_realm(self):
        c = RAGConfig()
        assert c.realm == "default"

    def test_default_gleaning_passes(self):
        c = RAGConfig()
        assert c.gleaning_passes == 1

    def test_default_chunk_chars(self):
        c = RAGConfig()
        assert c.chunk_chars == 2000
        assert c.chunk_overlap_chars == 200

    def test_default_concurrent_chunks(self):
        c = RAGConfig()
        assert c.max_concurrent_chunks == 4

    def test_default_community_min_size(self):
        c = RAGConfig()
        assert c.community_min_size == 3

    def test_default_booleans(self):
        c = RAGConfig()
        assert c.drop_negated_relations is False
        assert c.reject_possessive_entities is False
        assert c.skip_unchanged_documents is True
        assert c.exclude_dormant_entities is True
        assert c.extract_validity is True
        assert c.embed_relations is True
        assert c.expand_chunks_via_mentions is True
        assert c.include_superseded is False

    def test_default_relation_seed_quota(self):
        c = RAGConfig()
        assert c.relation_seed_quota == 0.5

    def test_default_encoding_format(self):
        c = RAGConfig()
        assert c.embedding_encoding_format == "float"

    def test_default_retry_config(self):
        c = RAGConfig()
        assert c.max_retries == 5
        assert c.retry_backoff_secs == 2.0
        assert c.retry_deadline_secs == 120


class TestRAGConfigOverrides:
    def test_explicit_values(self):
        c = RAGConfig(
            model="gpt-4", embedding_dim=768, max_hops=3,
            realm="my_realm", space="staging",
        )
        assert c.model == "gpt-4"
        assert c.embedding_dim == 768
        assert c.max_hops == 3
        assert c.realm == "my_realm"
        assert c.space == "staging"

    def test_fallback_models(self):
        c = RAGConfig(fallback_models=["model-a", "model-b"])
        assert c.fallback_models == ["model-a", "model-b"]

    def test_entity_types(self):
        c = RAGConfig(entity_types=["Person", "Org"])
        assert c.entity_types == ["Person", "Org"]

    def test_predicate_vocabulary(self):
        c = RAGConfig(predicate_vocabulary=["works_at", "knows"])
        assert c.predicate_vocabulary == ["works_at", "knows"]

    def test_predicate_aliases(self):
        c = RAGConfig(predicate_aliases={"employed_at": "works_at"})
        assert c.predicate_aliases == {"employed_at": "works_at"}

    def test_exclusive_predicate_groups(self):
        groups = [{"friend_of", "enemy_of"}]
        c = RAGConfig(exclusive_predicate_groups=groups)
        assert c.exclusive_predicate_groups == groups

    def test_community_weights(self):
        c = RAGConfig(
            community_weight_similarity=2.0,
            community_weight_importance=0.5,
            community_weight_size=0.1,
        )
        assert c.community_weight_similarity == 2.0
        assert c.community_weight_importance == 0.5
        assert c.community_weight_size == 0.1


class TestRAGConfigEnv:
    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("RAG_MODEL", "test-model-env")
        c = RAGConfig()
        assert c.model == "test-model-env"

    def test_env_embedding_dim(self, monkeypatch):
        monkeypatch.setenv("RAG_EMBEDDING_DIM", "768")
        c = RAGConfig()
        assert c.embedding_dim == 768

    def test_env_boolean_true(self, monkeypatch):
        monkeypatch.setenv("RAG_DROP_NEGATED", "1")
        c = RAGConfig()
        assert c.drop_negated_relations is True

    def test_env_boolean_false(self, monkeypatch):
        monkeypatch.setenv("RAG_DROP_NEGATED", "0")
        c = RAGConfig()
        assert c.drop_negated_relations is False

    def test_env_fallback_models(self, monkeypatch):
        monkeypatch.setenv("RAG_FALLBACK_MODELS", "model-a, model-b")
        c = RAGConfig()
        assert c.fallback_models == ["model-a", "model-b"]

    def test_env_fallback_models_empty(self, monkeypatch):
        monkeypatch.setenv("RAG_FALLBACK_MODELS", "")
        c = RAGConfig()
        assert c.fallback_models == []


class TestRAGConfigListDefaults:
    def test_entity_types_empty_by_default(self):
        c = RAGConfig()
        assert c.entity_types == []

    def test_predicate_vocabulary_empty_by_default(self):
        c = RAGConfig()
        assert c.predicate_vocabulary == []

    def test_predicate_aliases_empty_by_default(self):
        c = RAGConfig()
        assert c.predicate_aliases == {}

    def test_exclusive_predicate_groups_empty_by_default(self):
        c = RAGConfig()
        assert c.exclusive_predicate_groups == []

    def test_fallback_models_empty_by_default(self):
        c = RAGConfig()
        assert c.fallback_models == []
