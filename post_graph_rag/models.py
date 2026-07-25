"""Data models for post-graph-rag including DocumentMetadata."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

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
