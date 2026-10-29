"""A connection the caller supplies, so retrieval can reach a schema the
library's own credential may not.

The motivating shape: a tenant per PostgreSQL schema, where no single role is
permitted USAGE on every tenant's schema -- that reach being the thing
per-tenant schemas exist to remove. A worker connects as itself and assumes a
per-tenant role for the work. The library cannot do that from a DSN, because
the role is not a property of the connection string; it is a property of the
request.

Two things here are worth more than the happy path:

  * The control. Without the role, retrieval must *raise*, not return an
    empty result. An index that reads empty is indistinguishable from a
    corpus that holds nothing, and it is the worse failure.

  * The reason the advice is a session-level SET ROLE rather than SET LOCAL
    ROLE. Most read paths in post-graph do not open a transaction, so a
    transaction-local role set outside the library never reaches them. That
    is asserted below rather than left as a claim in a docstring.
"""

import uuid

import asyncpg
import pytest
from conftest import TEST_DSN, FakeLLM, _db_reachable, make_config

from post_graph_rag import GraphRAG


async def _grant_realm_to_role(admin, realm, role):
    """Everything the role needs on one realm schema, and nothing beyond it.

    Including USAGE on the schema that holds the `vector` extension, which is
    not optional and not obvious: pgvector's type lives there, so without it
    every vector read fails with `type "vector" does not exist` -- a message
    that names neither the role nor the schema actually missing. A database
    whose `public` has owner-only defaults (nspacl NULL, which is what
    `CREATE DATABASE` leaves on PostgreSQL 15+) gives a fresh role no USAGE
    on it at all, so this is the common case rather than the odd one.
    """
    ext_schema = await admin.fetchval(
        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
        "ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
    if ext_schema:
        await admin.execute(f'GRANT USAGE ON SCHEMA "{ext_schema}" TO "{role}"')
    await admin.execute(f'GRANT USAGE ON SCHEMA "{realm}" TO "{role}"')
    await admin.execute(
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{realm}" TO "{role}"')
    await admin.execute(
        f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "{realm}" TO "{role}"')


def _role_pool(role):
    """A pool that assumes *role* on every acquisition.

    Session-level on purpose: asyncpg runs `setup` after each acquire and
    `RESET ALL` on release, so the role covers every statement of the
    acquisition -- transactional or not -- and outlives none of it.
    """
    async def setup(conn):
        await conn.execute(f'SET ROLE "{role}"')
    return asyncpg.create_pool(TEST_DSN, setup=setup, min_size=1, max_size=4)


@pytest.fixture
def role_name():
    return f"seat_{uuid.uuid4().hex[:10]}"




async def _seeded_realm(answer="seeded"):
    """A realm with one indexed document, created by the admin credential."""
    config = make_config()
    if not await _db_reachable(config):
        pytest.skip("PostgreSQL with pgvector not reachable")
    rag = GraphRAG(config)
    rag.llm = FakeLLM(config, answer=answer)
    rag.extractor.llm_service = rag.llm
    await rag.initialize()
    await rag.index_document(
        "Vistra Corp reported refining operating costs of 412 million dollars.",
        metadata={"document": "transcript"},
    )
    return rag, config


class TestOwnership:
    """A library that closes a pool it merely borrowed would be worse than the
    problem the injection solves."""

    @pytest.mark.asyncio
    async def test_a_supplied_pool_is_not_closed_by_the_engine(self):
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        pool = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=2)
        try:
            rag = GraphRAG(config, connection=pool)
            rag.llm = FakeLLM(config)
            rag.extractor.llm_service = rag.llm
            await rag.initialize()
            await rag.close()
            assert not pool._closed, "the engine closed a pool it did not open"
            async with pool.acquire() as c:
                assert await c.fetchval("SELECT 1") == 1
        finally:
            await pool.close()
            try:
                admin = await asyncpg.connect(TEST_DSN)
                await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
                await admin.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_a_self_opened_pool_is_closed(self):
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        rag = GraphRAG(config)
        rag.llm = FakeLLM(config)
        rag.extractor.llm_service = rag.llm
        await rag.initialize()
        assert rag.store._owns_connection is True
        await rag.close()
        assert rag.store.client._pool._closed
        try:
            admin = await asyncpg.connect(TEST_DSN)
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await admin.close()
        except Exception:
            pass


class TestRetrievalUnderAnAssumedRole:
    """The point of the feature."""

    @pytest.mark.asyncio
    async def test_retrieval_reaches_the_realm_through_the_assumed_role(self, role_name):
        seeded, config = await _seeded_realm()
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)
            await seeded.close()

            pool = await _role_pool(role_name)
            try:
                rag = GraphRAG(config, connection=pool)
                rag.llm = FakeLLM(config, answer="from the case schema")
                rag.extractor.llm_service = rag.llm
                async with pool.acquire() as c:
                    assert await c.fetchval("SELECT current_user") == role_name

                # A real retrieval path, not just a count: search_similar_documents
                # is a pgvector read, and pgvector reads are the ones the report
                # says come back empty.
                hits = await rag.store.search_similar_documents(
                    [0.1] * config.embedding_dim, top_k=5)
                assert hits, "the seeded passage must be retrievable by the seat"
                docs = await rag.store.client.count_vertices("documents", realm=config.realm)
                assert docs >= 1
            finally:
                await pool.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()

    @pytest.mark.asyncio
    async def test_without_the_grant_retrieval_refuses_rather_than_reads_empty(self, role_name):
        """The control, and the more important half.

        A seat that silently retrieves nothing deliberates on an empty recall
        and looks healthy doing it. The failure has to be loud.
        """
        seeded, config = await _seeded_realm()
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')   # no grant on the realm
            await seeded.close()

            pool = await _role_pool(role_name)
            try:
                rag = GraphRAG(config, connection=pool)
                rag.llm = FakeLLM(config)
                rag.extractor.llm_service = rag.llm
                with pytest.raises(Exception) as exc:
                    await rag.store.client.count_vertices("documents", realm=config.realm)
                assert "permission denied" in str(exc.value).lower(), (
                    f"expected a refusal, got {exc.value!r}")
            finally:
                await pool.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()


class TestWhyTheAdviceIsSessionLevel:
    """SET LOCAL ROLE set outside the library does not reach its reads.

    This is the trap the docstring warns about, asserted rather than claimed:
    most read paths in post-graph issue their statement without opening a
    transaction, so a transaction-local role set beforehand is already gone.
    """

    @pytest.mark.asyncio
    async def test_a_transaction_local_role_does_not_survive_to_a_read(self, role_name):
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        conn = await asyncpg.connect(TEST_DSN)
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            # SET LOCAL ROLE outside any transaction is released immediately:
            # the implicit transaction that took it ends with the statement.
            await conn.execute("SELECT set_config('role', $1, true)", role_name)
            assert await conn.fetchval("SELECT current_user") != role_name, (
                "a transaction-local role outlived its implicit transaction")

            # Session level is what a later, separate statement still sees.
            await conn.execute(f'SET ROLE "{role_name}"')
            assert await conn.fetchval("SELECT current_user") == role_name
            await conn.execute("RESET ROLE")
        finally:
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await conn.close()
            await admin.close()


class TestTheVectorExtensionGrant:
    """The grant that is easy to miss and fails in a misleading way."""

    @pytest.mark.asyncio
    async def test_without_usage_on_the_extension_schema_vector_reads_fail(self, role_name):
        """A role granted everything on its own realm, and nothing on the
        schema holding pgvector, cannot do a vector read at all.

        The message names the type, not the missing privilege or the schema
        it lives in, so this is worth a test that states the cause once.
        """
        seeded, config = await _seeded_realm()
        admin = await asyncpg.connect(TEST_DSN)
        try:
            ext_schema = await admin.fetchval(
                "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
                "ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
            if not ext_schema:
                pytest.skip("vector extension not installed")
            if await admin.fetchval(
                    "SELECT has_schema_privilege('public', $1, 'USAGE')", ext_schema):
                pytest.skip(
                    f"PUBLIC already holds USAGE on {ext_schema!r}, so the "
                    "omission this test describes cannot arise here")

            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            # Deliberately NOT granting USAGE on the extension schema.
            await admin.execute(f'GRANT USAGE ON SCHEMA "{config.realm}" TO "{role_name}"')
            await admin.execute(
                f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA '
                f'"{config.realm}" TO "{role_name}"')
            await seeded.close()

            pool = await _role_pool(role_name)
            try:
                rag = GraphRAG(config, connection=pool)
                rag.llm = FakeLLM(config)
                rag.extractor.llm_service = rag.llm

                # A non-vector read works: the realm grant is fine.
                assert await rag.store.client.count_vertices(
                    "documents", realm=config.realm) >= 1

                # The vector read does not.
                with pytest.raises(Exception) as exc:
                    await rag.store.search_similar_documents(
                        [0.1] * config.embedding_dim, top_k=5)
                assert 'type "vector" does not exist' in str(exc.value), (
                    f"expected the misleading vector error, got {exc.value!r}")
            finally:
                await pool.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()


class TestTheWholeSeatPath:
    """Index and retrieve, both through the assumed role.

    The report describes a seat whose reads *and* writes arrive as the wrong
    role, so a read-only proof would be half a proof.
    """

    @pytest.mark.asyncio
    async def test_a_document_indexed_and_retrieved_under_the_role(self, role_name):
        seeded, config = await _seeded_realm()
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)
            await seeded.close()

            pool = await _role_pool(role_name)
            try:
                rag = GraphRAG(config, connection=pool)
                rag.llm = FakeLLM(config, answer="under the role")
                rag.extractor.llm_service = rag.llm

                before = await rag.store.client.count_vertices("documents", realm=config.realm)
                await rag.index_document(
                    "A second passage, written by the seat itself.",
                    metadata={"document": "seat-written"},
                )
                after = await rag.store.client.count_vertices("documents", realm=config.realm)
                assert after > before, "the write did not land under the assumed role"

                hits = await rag.store.search_similar_documents(
                    [0.1] * config.embedding_dim, top_k=10)
                assert hits, "what the seat wrote must be what the seat can recall"

                # And it really was the role, not the connecting user.
                async with pool.acquire() as c:
                    assert await c.fetchval("SELECT current_user") == role_name
                    assert await c.fetchval("SELECT session_user") != role_name
            finally:
                await pool.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()


class TestASuppliedPoolNeedNotBeAnAsyncpgPool:
    """A *wrapper* around a pool is the shape that works behind a pooler.

    `GraphRAG(config, connection=...)` hands whatever it is given to
    `AsyncPostGraph`, which used to decide pool-or-connection by
    `isinstance(..., asyncpg.Pool)`. Anything else was driven as a single
    connection, `acquire()` was never called, and the first statement died on
    `AttributeError: 'X' object has no attribute 'fetch'`.

    The object that falls through that test is exactly the useful one. A
    session-level `SET ROLE` is unsafe through PgBouncer in transaction mode --
    the server connection keeps the role for whoever borrows it next -- so the
    role has to be assumed per acquisition, inside a transaction, by a wrapper.
    Subclassing `asyncpg.Pool` to be recognised would oblige the caller to
    import asyncpg, which is what a caller routing all its SQL through this
    library is trying to avoid.
    """

    class _TxRolePool:
        """Wraps a pool; each acquisition is a transaction wearing *role*.

        Presents `acquire()` and nothing else -- no `fetch`, no `execute` -- so
        any site that mistakes it for a connection raises rather than quietly
        working on the wrong credential.
        """

        def __init__(self, pool, role):
            self._pool = pool
            self._role = role
            self.acquisitions = 0

        def acquire(self):
            outer = self

            class _Ctx:
                async def __aenter__(self):
                    outer.acquisitions += 1
                    self._cm = outer._pool.acquire()
                    self._conn = await self._cm.__aenter__()
                    self._tx = self._conn.transaction()
                    await self._tx.start()
                    await self._conn.execute(f'SET LOCAL ROLE "{outer._role}"')
                    return self._conn

                async def __aexit__(self, exc_type, exc, tb):
                    if exc_type is None:
                        await self._tx.commit()
                    else:
                        await self._tx.rollback()
                    return await self._cm.__aexit__(exc_type, exc, tb)

            return _Ctx()

    @pytest.mark.asyncio
    async def test_retrieval_runs_through_a_pool_wrapper(self, role_name):
        seeded, config = await _seeded_realm(answer="through the wrapper")
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)
            await seeded.close()

            raw = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=4)
            try:
                wrapper = self._TxRolePool(raw, role_name)
                rag = GraphRAG(config, connection=wrapper)
                rag.llm = FakeLLM(config, answer="through the wrapper")
                rag.extractor.llm_service = rag.llm

                # The read path: no transaction of its own, so this is the one
                # the wrapper has to cover.
                hits = await rag.store.search_similar_documents(
                    [0.1] * config.embedding_dim, top_k=5)
                assert hits, "the seeded passage must be retrievable"

                # The write path, which goes through _run_in_tx.
                await rag.store.client.upsert_vertex(
                    "documents", realm=config.realm, vertex_id=999999,
                    payload={"text": "written through the wrapper"})

                # And the delta reader, which had its own copy of the
                # pool-or-connection decision and the same mistake in it.
                from post_graph_rag.deltas import DeltaReader
                as_of = await DeltaReader(rag.store)._db_now()
                assert as_of

                assert wrapper.acquisitions >= 3
            finally:
                await raw.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()

    @pytest.mark.asyncio
    async def test_the_wrapper_really_carried_the_role(self, role_name):
        """The control: without the grant the same wrapper is refused.

        Otherwise the test above would pass just as well if the wrapper's
        `SET LOCAL ROLE` never ran.
        """
        seeded, config = await _seeded_realm()
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')  # no grant
            await seeded.close()

            raw = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=4)
            try:
                rag = GraphRAG(config,
                               connection=self._TxRolePool(raw, role_name))
                rag.llm = FakeLLM(config)
                rag.extractor.llm_service = rag.llm
                with pytest.raises(Exception) as exc:
                    await rag.store.client.count_vertices("documents",
                                                          realm=config.realm)
                assert "permission denied" in str(exc.value).lower(), (
                    f"expected a refusal, got {exc.value!r}")
            finally:
                await raw.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()
