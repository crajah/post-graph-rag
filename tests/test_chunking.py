"""Tests for post_graph_rag.chunking — paragraph chunker edge cases."""

from post_graph_rag.chunking import (
    paragraph_chunker,
    make_paragraph_chunker,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
)


class TestParagraphChunkerBasics:
    def test_short_text_single_chunk(self):
        text = "This is a short paragraph with enough chars to pass the minimum threshold."
        chunks = paragraph_chunker(text)
        assert len(chunks) == 1
        assert chunks[0].strip() == text.strip()

    def test_empty_text(self):
        assert paragraph_chunker("") == []

    def test_whitespace_only(self):
        assert paragraph_chunker("   \n\n\n   ") == []

    def test_headings_skipped_by_default(self):
        text = "== Section Heading ==\n\nThis is a real paragraph with enough characters to be included."
        chunks = paragraph_chunker(text)
        assert len(chunks) == 1
        assert "Section Heading" not in chunks[0]

    def test_headings_kept_when_disabled(self):
        text = "== Section Heading == with lots of extra text to meet the minimum\n\nAnother real paragraph that is long enough to be kept by the chunker."
        chunks = paragraph_chunker(text, skip_headings=False)
        found = any("Section Heading" in c for c in chunks)
        assert found

    def test_short_paragraphs_skipped(self):
        text = "Short.\n\nThis paragraph is definitely long enough to be included by the default chunker settings."
        chunks = paragraph_chunker(text)
        assert len(chunks) == 1
        assert "Short." not in chunks[0]

    def test_custom_min_paragraph_chars(self):
        text = "Short.\n\nAlso short but longer."
        chunks = paragraph_chunker(text, min_paragraph_chars=5)
        assert any("Also short" in c for c in chunks)

    def test_multiple_chunks_created(self):
        para = "A" * 100 + " is a word. " + "B" * 100
        text = "\n\n".join([para] * 20)
        chunks = paragraph_chunker(text, chunk_chars=500)
        assert len(chunks) > 1

    def test_overlap_carried_forward(self):
        para_a = "First paragraph. " * 10  # ~170 chars
        para_b = "Second paragraph. " * 10
        para_c = "Third paragraph. " * 10
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"
        chunks = paragraph_chunker(text, chunk_chars=200, overlap_chars=50)
        if len(chunks) > 1:
            # Assert the overlap, rather than that the next chunk is non-empty.
            # The original computed the tail and then checked only len() > 0,
            # which passes for any chunker at all — including one with no
            # overlap, which is the thing this test exists to catch.
            tail_of_first = chunks[0][-50:].strip()
            assert any(word and word in chunks[1] for word in tail_of_first.split()), (
                "no content from the first chunk carried into the second")

    def test_no_overlap(self):
        para = "X" * 60
        text = "\n\n".join([para] * 10)
        chunks = paragraph_chunker(text, chunk_chars=100, overlap_chars=0)
        assert len(chunks) > 1


class TestMakeParagraphChunker:
    def test_returns_callable(self):
        chunker = make_paragraph_chunker()
        assert callable(chunker)

    def test_uses_provided_params(self):
        chunker = make_paragraph_chunker(chunk_chars=100, overlap_chars=10)
        para = "A" * 60
        text = "\n\n".join([para] * 5)
        chunks = chunker(text)
        assert len(chunks) > 1

    def test_matches_direct_call(self):
        text = "This is a decent paragraph with enough content to be meaningful for chunking.\n\nAnd another paragraph here."
        direct = paragraph_chunker(text, chunk_chars=2000, overlap_chars=200)
        via_factory = make_paragraph_chunker(chunk_chars=2000, overlap_chars=200)(text)
        assert direct == via_factory


class TestChunkerConstants:
    def test_default_chunk_chars(self):
        assert DEFAULT_CHUNK_CHARS == 2000

    def test_default_overlap_chars(self):
        assert DEFAULT_OVERLAP_CHARS == 200
