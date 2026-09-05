"""Belief time must govern the document channel, not only the relations.

Filtering relations alone left post-watermark text in the answer, so "what
did we believe in March" could quote an April restatement -- the audit
guarantee failing exactly where it is relied on.

Chunks are written through the store directly: extraction is not what is
under test here, and the fake extractor would refuse this text anyway.
"""
import asyncio
from datetime import datetime, timezone

import pytest
from conftest import VOCAB_DIM

from post_graph_rag import QueryParam

pytestmark = pytest.mark.asyncio

EMB = [0.1] * VOCAB_DIM


class TestBeliefTimeDocuments:
    async def test_chunks_after_watermark_are_withheld(self, rag_factory):
        rag = await rag_factory()
        await rag.store.add_document("Calder reported 2019 revenue of 412 million.",
                                     embedding=EMB, metadata={"source": "a.txt"})
        watermark = datetime.now(timezone.utc).isoformat()
        await asyncio.sleep(1.1)
        await rag.store.add_document("Calder restated 2019 revenue to 388 million.",
                                     embedding=EMB, metadata={"source": "b.txt"})

        now = await rag.query_data("2019 revenue?", param=QueryParam())
        past = await rag.query_data(
            "2019 revenue?", param=QueryParam(as_believed_at=watermark))
        assert len(now["data"]["chunks"]) == 2
        assert len(past["data"]["chunks"]) == 1
        assert "412" in (past["data"]["chunks"][0]["content"] or "")

    async def test_no_watermark_returns_everything(self, rag_factory):
        rag = await rag_factory()
        await rag.store.add_document("Alpha reported growth.", embedding=EMB,
                                     metadata={"source": "a.txt"})
        await rag.store.add_document("Beta reported growth.", embedding=EMB,
                                     metadata={"source": "b.txt"})
        out = await rag.query_data("growth?", param=QueryParam())
        assert len(out["data"]["chunks"]) == 2

    async def test_unparseable_watermark_does_not_drop_evidence(self, rag_factory):
        rag = await rag_factory()
        await rag.store.add_document("Gamma reported growth.", embedding=EMB,
                                     metadata={"source": "a.txt"})
        out = await rag.query_data("growth?",
                                   param=QueryParam(as_believed_at="not-a-date"))
        # a bad watermark must not silently empty the context
        assert len(out["data"]["chunks"]) == 1
