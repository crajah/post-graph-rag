"""Tests for post_graph_rag package-level exports."""

import post_graph_rag


class TestPackageExports:
    def test_version_is_string(self):
        assert isinstance(post_graph_rag.__version__, str)
        assert "." in post_graph_rag.__version__

    def test_all_list_exists(self):
        assert isinstance(post_graph_rag.__all__, list)
        assert len(post_graph_rag.__all__) > 0

    def test_all_entries_are_importable(self):
        for name in post_graph_rag.__all__:
            assert hasattr(post_graph_rag, name), f"{name} listed in __all__ but not importable"

    def test_core_classes_exported(self):
        assert hasattr(post_graph_rag, "RAGConfig")
        assert hasattr(post_graph_rag, "GraphRAG")
        assert hasattr(post_graph_rag, "RAGGraphStore")
        assert hasattr(post_graph_rag, "LLMService")
        assert hasattr(post_graph_rag, "GraphExtractor")

    def test_model_classes_exported(self):
        assert hasattr(post_graph_rag, "DocumentContext")
        assert hasattr(post_graph_rag, "DocumentMetadata")
        assert hasattr(post_graph_rag, "QueryParam")
        assert hasattr(post_graph_rag, "KeywordResult")

    def test_error_classes_exported(self):
        assert hasattr(post_graph_rag, "RAGError")
        assert hasattr(post_graph_rag, "SchemaError")
        assert hasattr(post_graph_rag, "EmbeddingError")
        assert hasattr(post_graph_rag, "LLMError")
        assert hasattr(post_graph_rag, "ExtractionError")

    def test_extraction_classes_exported(self):
        assert hasattr(post_graph_rag, "Entity")
        assert hasattr(post_graph_rag, "Triple")
        assert hasattr(post_graph_rag, "ExtractionResult")

    def test_chunking_exported(self):
        assert hasattr(post_graph_rag, "Chunker")
        assert hasattr(post_graph_rag, "paragraph_chunker")
        assert hasattr(post_graph_rag, "make_paragraph_chunker")

    def test_community_exports(self):
        assert hasattr(post_graph_rag, "CommunityDetector")
        assert hasattr(post_graph_rag, "default_detector")
        assert hasattr(post_graph_rag, "label_propagation")
        assert hasattr(post_graph_rag, "group_by_community")

    def test_reporting_exports(self):
        assert hasattr(post_graph_rag, "CommunityReport")
        assert hasattr(post_graph_rag, "CommunityReporter")
        assert hasattr(post_graph_rag, "Finding")
