"""Data models for post-graph-rag including DocumentMetadata, QueryParam, and KeywordResult."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class DocumentMetadata:
    """Structured document metadata for knowledge graph indexing and retrieval.
    
    All fields are optional to accommodate unstructured strings, snippets, and structured files.
    """
    source: Optional[str] = None        # e.g., URL, file path, API source
    category: Optional[str] = None      # e.g., "manuals", "contracts", "research"
    collection: Optional[str] = None    # e.g., "engineering_wiki", "q3_reports"
    document: Optional[str] = None      # e.g., "architecture_spec.pdf", "user_guide.md"
    page: Optional[int] = None          # e.g., Page number (1-based)
    paragraph: Optional[int] = None     # e.g., Paragraph index (1-based)
    extra: Dict[str, Any] = field(default_factory=dict) # Any additional custom key-value pairs

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentMetadata to dictionary representation, omitting None values."""
        res = {
            "source": self.source,
            "category": self.category,
            "collection": self.collection,
            "document": self.document,
            "page": self.page,
            "paragraph": self.paragraph,
        }
        if self.extra:
            res.update(self.extra)
        return {k: v for k, v in res.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Reconstruct DocumentMetadata from dictionary data."""
        if not data:
            return cls()
        known_keys = {"source", "category", "collection", "document", "page", "paragraph"}
        known_args = {k: data[k] for k in known_keys if k in data}
        extra_args = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**known_args, extra=extra_args)

@dataclass
class KeywordResult:
    """Dual-level keyword extraction result."""
    high_level_keywords: List[str] = field(default_factory=list)
    low_level_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_level": self.high_level_keywords,
            "low_level": self.low_level_keywords
        }

@dataclass
class QueryParam:
    """Configuration parameters controlling RAG retrieval and synthesis behavior."""
    mode: str = "mix"                           # "mix", "local", "global", "hybrid", "naive", or "bypass"
    top_k: int = 5                              # Top items to retrieve
    max_total_tokens: int = 4000                # Total context token budget
    max_entity_tokens: int = 1500               # Max entity context token budget
    max_relation_tokens: int = 1500             # Max relation context token budget
    response_type: str = "Multiple Paragraphs"  # Expected answer formatting
    stream: bool = False                        # Stream response chunks if True
    only_need_context: bool = False             # Return raw context without synthesis if True
    conversation_history: List[Dict[str, str]] = field(default_factory=list) # Multi-turn chat history
    hl_keywords: List[str] = field(default_factory=list) # Custom high-level search terms
    ll_keywords: List[str] = field(default_factory=list) # Custom low-level search terms
