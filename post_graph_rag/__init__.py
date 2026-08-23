"""post-graph-rag: Graph RAG library using post-graph and pgvector on PostgreSQL."""

from post_graph_rag.config import RAGConfig
from post_graph_rag.errors import (
    RAGError,
    SchemaError,
    EmbeddingError,
    LLMError,
    ExtractionError,
)
from post_graph_rag.chunking import Chunker, make_paragraph_chunker, paragraph_chunker
from post_graph_rag.communities import (
    CommunityDetector, default_detector, group_by_community, label_propagation,
)
from post_graph_rag.reporting import CommunityReport, CommunityReporter, Finding
from post_graph_rag.models import DocumentContext, DocumentMetadata, QueryParam, KeywordResult
from post_graph_rag.llm import LLMService
from post_graph_rag.extractor import GraphExtractor, Entity, Triple, ExtractionResult
from post_graph_rag.graph_store import RAGGraphStore
from post_graph_rag.engine import GraphRAG

__version__ = "1.8.0"
__all__ = [
    "RAGConfig",
    "RAGError",
    "SchemaError",
    "EmbeddingError",
    "LLMError",
    "ExtractionError",
    "DocumentContext",
    "DocumentMetadata",
    "QueryParam",
    "KeywordResult",
    "Chunker",
    "paragraph_chunker",
    "make_paragraph_chunker",
    "CommunityDetector",
    "default_detector",
    "label_propagation",
    "group_by_community",
    "CommunityReport",
    "CommunityReporter",
    "Finding",
    "LLMService",
    "GraphExtractor",
    "Entity",
    "Triple",
    "ExtractionResult",
    "RAGGraphStore",
    "GraphRAG"
]
