"""Data models for post-graph-rag including DocumentMetadata, QueryParam, and KeywordResult."""
import hashlib
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
    space: Optional[str] = None         # e.g., Sub-grouping space ("production", "sandbox")

    # Identity, used to recognise a document across re-indexing runs. Both are
    # normally set by the engine, but are first-class fields rather than `extra`
    # entries so callers can read and compare them directly.
    doc_key: Optional[str] = None       # Stable document identity (source, else title)
    content_hash: Optional[str] = None  # Digest of this chunk's text; differs iff the text changed

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
            "space": self.space,
            "doc_key": self.doc_key,
            "content_hash": self.content_hash,
        }
        if self.extra:
            res.update(self.extra)
        return {k: v for k, v in res.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Reconstruct DocumentMetadata from dictionary data."""
        if not data:
            return cls()
        known_keys = {"source", "category", "collection", "document", "page", "paragraph",
                      "space", "doc_key", "content_hash"}
        known_args = {k: data[k] for k in known_keys if k in data}
        extra_args = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**known_args, extra=extra_args)

@dataclass
class DocumentContext:
    """Context supplied to extraction alongside a chunk.

    A chunk taken from the middle of a document is full of references that only
    resolve against what came before it ("he", "the engine", "his father").
    Extracting it blind produces unresolvable vertices and misattributed
    relations, so callers pass what is known about the surrounding document.
    """
    title: Optional[str] = None          # e.g. "Charles Babbage"
    source: Optional[str] = None         # e.g. the URL or file path
    summary: Optional[str] = None        # running summary of preceding chunks
    known_entities: List[str] = field(default_factory=list)  # canonical names seen so far

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "summary": self.summary,
            "known_entities": self.known_entities,
        }


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

def document_key(source: Optional[str], document: Optional[str]) -> str:
    """Stable identity for a document across re-indexing runs.

    Both parts are used. Preferring the source alone was a trap: a caller that
    passes a constant source — a corpus name rather than a path — collapses every
    document onto one key, and since a matching key means *re-index*, each
    document deletes the one before it. That failed silently in an ECT-QA run,
    where 80 transcripts shared ``source="ect"`` and the graph ended up holding
    only the last quarter of each company, 92% of its relations dormant.

    Combining them costs the case where a file is renamed but keeps its path:
    the title moves, so the key moves, and the re-index appends instead of
    replacing. That is a duplicate — visible, and recoverable by deleting it.
    The alternative failure mode destroys data and looks like nothing happened.
    """
    src = (source or "").strip()
    doc = (document or "").strip()
    if src and doc:
        return f"{src}::{doc}"
    return src or doc or "unkeyed"


def content_hash(text: str) -> str:
    """Digest used to tell an unchanged chunk from an edited one."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


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
    space: Optional[str] = None                 # Optional space filter ("production", "sandbox")
    as_of: Optional[str] = None                 # Only relations VALID at this date ("1625", "1625-06-12")
    # The second temporal axis: the graph as this system believed it at an
    # instant, rather than the world as it was. A filing published in 2024 can
    # assert something true in 2019, so `as_of` and `as_believed_at` select
    # different rows and answer different questions — "what was true then"
    # against "what did we know then". The latter is what reproduces a past
    # answer or audits a decision.
    as_believed_at: Optional[str] = None        # Graph as known at this ISO instant
    include_superseded: Optional[bool] = None   # Include relations a later assertion replaced

    # Additional retrieval texts whose candidates merge into the same pools
    # before ranking. A question comparing two events embeds as one vector that
    # resembles neither event, so the second event's facts sit below the cutoff
    # however large top_k is. Retrieving each event separately is the fix; the
    # fusion step already knows how to rank candidates from several sources.
    subqueries: Optional[List[str]] = None
    max_hops: Optional[int] = None              # Hops to walk from a matched entity (None = config default)
    conversation_history: List[Dict[str, str]] = field(default_factory=list) # Multi-turn chat history
    hl_keywords: List[str] = field(default_factory=list) # Custom high-level search terms
    ll_keywords: List[str] = field(default_factory=list) # Custom low-level search terms


@dataclass
class CommunityCoverage:
    """How much retrieval attention a community has received."""
    community_id: str
    title: str
    members: int
    retrieval_hits: int
    last_hit_at: Optional[str]
    hit_share: float          # hits normalised by member count
