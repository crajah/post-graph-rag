"""changes_since: what changed after T, answered from belief time.

Seeded through the store API (no LLM), so every stamp under test is the one
production writes. The watermark discipline is the core assertion: polling
with each delta's `as_of` must see every change exactly once.
"""
import asyncio

import pytest

pytestmark = pytest.mark.asyncio

EMB = [0.1] * 8


async def _ent(rag, name):
    return await rag.store.upsert_entity(name=name, entity_type="Thing",
                                         description=f"{name} desc", embedding=EMB)


class TestChangesSince:
    async def test_new_relation_appears_once_across_watermarks(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        a, b = await _ent(rag, "Alpha"), await _ent(rag, "Beta")
        d0 = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        base = d0.as_of

        await rag.store.add_relation(a, b, "works_with", description="w")
        d1 = await rag.changes_since(base, summary=False)
        assert d1.counts["new_relations"] == 1
        assert len(d1.new_relations) == 1
        assert d1.new_relations[0].relation_type == "works_with"

        # Next poll from the new watermark: nothing, exactly once semantics.
        d2 = await rag.changes_since(d1.as_of, summary=True)
        assert d2.counts["new_relations"] == 0

    async def test_supersession_lands_in_superseded_bucket(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        a, b = await _ent(rag, "Gamma"), await _ent(rag, "Delta")
        await rag.store.add_relation(a, b, "ally_of", description="then")
        d0 = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)

        # Supersession is decided by the indexing engine; at store level the
        # contract is supersede_conflicting, which is what stamps t_expired.
        new_edge = await rag.store.add_relation(a, b, "enemy_of", description="now")
        await rag.store.supersede_conflicting(
            from_id=a.id, to_id=b.id, new_predicate="enemy_of",
            new_edge_id=new_edge.id,
            exclusive_groups=[{"ally_of", "enemy_of"}])
        d1 = await rag.changes_since(d0.as_of, summary=False)
        assert d1.counts["new_relations"] == 1          # enemy_of
        assert d1.counts["superseded_relations"] == 1   # ally_of closed
        sup = d1.superseded_relations[0]
        assert sup.relation_type == "ally_of"
        assert sup.payload.get("superseded_by") is not None
        assert sup.payload.get("t_expired") is not None

    async def test_new_entities_and_documents(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        d0 = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        await _ent(rag, "Epsilon")
        await rag.store.add_document("some text", embedding=EMB,
                                     metadata={"document": "doc-one", "source": "s"})
        d1 = await rag.changes_since(d0.as_of, summary=False)
        assert d1.counts["new_entities"] == 1
        assert d1.new_entities[0]["payload"].get("name", "").lower() == "epsilon" or \
               "epsilon" in str(d1.new_entities[0]["payload"]).lower()
        assert d1.counts["new_documents"] == 1
        assert d1.new_documents[0]["chunks"] == 1

    async def test_summary_counts_match_detail(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        a, b = await _ent(rag, "Zeta"), await _ent(rag, "Eta")
        d0 = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        await rag.store.add_relation(a, b, "knows", description="k")
        s = await rag.changes_since(d0.as_of, summary=True)
        f = await rag.changes_since(d0.as_of, summary=False)
        assert s.counts["new_relations"] == len(f.new_relations) == 1
        assert s.new_relations == []                     # summary transfers no rows

    async def test_empty_delta_is_empty(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        await _ent(rag, "Theta")
        d0 = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        d1 = await rag.changes_since(d0.as_of, summary=True)
        assert d1.empty, d1.counts

    async def test_communities_stale_flag(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        a, b = await _ent(rag, "Iota"), await _ent(rag, "Kappa")
        await rag.store.add_relation(a, b, "linked_to", description="l")
        d = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        # No communities built at all: nothing can be stale.
        assert d.communities_stale is False

    async def test_include_scoping_and_bad_include(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        d = await rag.changes_since("1970-01-01T00:00:00+00:00",
                                    include=("relations",), summary=True)
        assert "new_entities" not in d.counts
        with pytest.raises(ValueError, match="Unknown include"):
            await rag.changes_since("1970-01-01T00:00:00+00:00", include=("bogus",))

    async def test_watermark_is_utc_isoformat(self, rag_factory):
        rag = await rag_factory(embedding_dim=8)
        d = await rag.changes_since("1970-01-01T00:00:00+00:00", summary=True)
        assert "+00:00" in d.as_of or d.as_of.endswith("Z")
