"""Deleting one document, including the copies the audit trail kept.

`documents` and `doc_mentions` are created audited, so every chunk is held
three times: once live, once under `new_row` from the INSERT that wrote it, and
once under `old_row` from the DELETE that removed it -- with the chunk text and
its embedding in each. For a public reference shelf that is the point. For a
document somebody asked to have deleted, it is the whole of the problem: the
live row goes and every word of it stays.

Dropping the realm clears it, but a realm is everything a person has, not the
one document they deleted. And this method is the only place that can do it:
once it returns, the ids are gone from the live tables.

`purge_audit` is therefore off by default and additive in the receipt, so a
re-indexing caller sees no change at all.
"""

import uuid

import pytest
from conftest import FakeLLM, _db_reachable, fake_embed, make_config

from post_graph_rag import GraphRAG
from post_graph_rag.graph_store import RAGGraphStore


async def _store_with_two_documents(secret: str, keep: str):
    """A realm holding a document to delete and one to leave alone."""
    config = make_config()
    if not await _db_reachable(config):
        pytest.skip("PostgreSQL with pgvector not reachable")
    rag = GraphRAG(config)
    rag.llm = FakeLLM(config, answer="x")
    rag.extractor.llm_service = rag.llm
    await rag.initialize()

    store = rag.store
    secret_doc = await store.add_document(
        secret, fake_embed(secret), metadata={"doc_key": "diary-secret"})
    await store.add_document(
        keep, fake_embed(keep), metadata={"doc_key": "notes-keep"})
    # A mention edge from the secret chunk, so the edge audit rows exist too.
    entity = await store.upsert_entity("zeus", "deity", "the sky",
                                       fake_embed("zeus"))
    await store.client.add_edge(
        "doc_mentions", realm=config.realm, from_id=secret_doc.id,
        to_id=entity.id, relation_type="mentions", payload={})
    return rag, config


class TestThePurge:
    """The reason the flag exists."""

    @pytest.mark.asyncio
    async def test_the_deleted_document_leaves_no_trace(self):
        secret = f"SECRET_CONFESSION_{uuid.uuid4().hex[:8]}"
        keep = f"ORDINARY_NOTES_{uuid.uuid4().hex[:8]}"
        rag, config = await _store_with_two_documents(secret, keep)
        realm = config.realm
        try:
            # Before: the text is in the live table and in the audit copy.
            assert await rag.store.client.tables_containing_text(
                secret, include_realms=[realm]) == {
                    f"{realm}.documents", f"{realm}.documents_audit"}

            receipt = await rag.store.delete_document_chunks(
                "diary-secret", space=config.space, purge_audit=True)

            assert receipt["chunks"] >= 1
            assert receipt["mentions"] == 1
            # INSERT and DELETE audit rows for the chunk, and the same pair for
            # the mention edge, at least.
            assert receipt["audit_rows_purged"] >= 4, receipt

            # Nothing left anywhere: not the live row, not the shadow data
            # table, not either half of the audit copy.
            assert await rag.store.client.tables_containing_text(
                secret, include_realms=[realm]) == set()

            # And the document nobody asked about is untouched, history included.
            assert await rag.store.client.tables_containing_text(
                keep, include_realms=[realm]) == {
                    f"{realm}.documents", f"{realm}.documents_audit"}
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            await rag.close()

    @pytest.mark.asyncio
    async def test_the_default_keeps_the_history(self):
        """The control, and the behaviour every existing caller gets.

        Without this the test above would prove only that deletion works.
        """
        secret = f"SECRET_CONFESSION_{uuid.uuid4().hex[:8]}"
        keep = f"ORDINARY_NOTES_{uuid.uuid4().hex[:8]}"
        rag, config = await _store_with_two_documents(secret, keep)
        realm = config.realm
        try:
            receipt = await rag.store.delete_document_chunks(
                "notes-keep", space=config.space)

            assert receipt["chunks"] >= 1
            assert "audit_rows_purged" not in receipt, (
                "the receipt must not grow a key for callers that did not ask")
            assert set(receipt) == {"chunks", "mentions", "dormant",
                                    "relations_withdrawn", "relations_orphaned"}

            # Live row gone, audit history kept.
            hits = await rag.store.client.tables_containing_text(
                keep, include_realms=[realm])
            assert hits == {f"{realm}.documents_audit"}, hits
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            await rag.close()

    @pytest.mark.asyncio
    async def test_the_purge_runs_after_the_deletes(self):
        """Ordering, asserted rather than left to the comment.

        The DELETEs are audited too, so purging first would leave the row each
        one writes on its way out -- the copy with the full text in old_row.
        A purge that ran first would leave exactly one audit row behind.
        """
        secret = f"SECRET_ORDER_{uuid.uuid4().hex[:8]}"
        rag, config = await _store_with_two_documents(secret, "other text here")
        realm = config.realm
        try:
            await rag.store.delete_document_chunks(
                "diary-secret", space=config.space, purge_audit=True)
            rows = await rag.store.client._fetch(
                f'SELECT action FROM "{realm}"."documents_audit"')
            actions = [r["action"] for r in rows]
            assert "DELETE" not in actions, (
                f"a DELETE audit row survived the purge, so it ran too early: "
                f"{actions}")
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            await rag.close()

    @pytest.mark.asyncio
    async def test_entity_and_relation_history_is_left_alone(self):
        """Entities are marked dormant rather than deleted, and both they and
        relations are shared: another document in the space may have
        contributed the same assertion."""
        secret = f"SECRET_SHARED_{uuid.uuid4().hex[:8]}"
        rag, config = await _store_with_two_documents(secret, "other text here")
        realm = config.realm
        try:
            before = await rag.store.client._fetchrow(
                f'SELECT count(*) AS n FROM "{realm}"."entities_audit"')
            await rag.store.delete_document_chunks(
                "diary-secret", space=config.space, purge_audit=True)
            after = await rag.store.client._fetchrow(
                f'SELECT count(*) AS n FROM "{realm}"."entities_audit"')
            assert after["n"] >= before["n"], (
                "entity audit rows must not be purged")
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            await rag.close()

    @pytest.mark.asyncio
    async def test_an_absent_document_reports_zero(self):
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        store = RAGGraphStore(config)
        await store.connect()
        try:
            await store.initialize_schema()
            receipt = await store.delete_document_chunks(
                "no-such-document", purge_audit=True)
            assert receipt["chunks"] == 0
            assert receipt["audit_rows_purged"] == 0
            plain = await store.delete_document_chunks("no-such-document")
            assert "audit_rows_purged" not in plain
        finally:
            await store.client._execute(
                f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await store.close()
