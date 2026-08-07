"""Text chunking for indexing.

The library previously left chunking entirely to the caller, which meant every
user reimplemented it and relations spanning a boundary were silently lost. This
provides a reasonable default with overlap, and a protocol so callers can supply
their own splitter without forking anything.
"""
from typing import Callable, List, Protocol

DEFAULT_CHUNK_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200


class Chunker(Protocol):
    """Any callable that splits a document into indexable chunks."""

    def __call__(self, text: str) -> List[str]:  # pragma: no cover - structural type
        ...


def paragraph_chunker(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_paragraph_chars: int = 40,
    skip_headings: bool = True,
) -> List[str]:
    """Pack paragraphs up to ``chunk_chars``, carrying ``overlap_chars`` forward.

    Overlap matters for graph extraction specifically: a relation whose subject
    appears in the last sentence of one chunk and whose object appears in the
    first sentence of the next is invisible to both without it.
    """
    paragraphs = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if skip_headings and para.startswith("=="):
            continue
        if len(para) < min_paragraph_chars:
            continue
        paragraphs.append(para)

    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > chunk_chars:
            chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            # Resume at a sentence boundary so the overlap reads as prose.
            if tail:
                cut = tail.find(". ")
                tail = tail[cut + 2:] if 0 <= cut < len(tail) - 2 else tail
            buf = f"{tail}\n{para}".strip() if tail else para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def make_paragraph_chunker(
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> Callable[[str], List[str]]:
    """Build a configured :func:`paragraph_chunker`."""
    def _chunk(text: str) -> List[str]:
        return paragraph_chunker(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    return _chunk
