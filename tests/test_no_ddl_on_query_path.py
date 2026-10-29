"""No DDL on the query path, so a role that does not own the tables can retrieve.

The deployment shape this is for: each tenant is a PostgreSQL schema, the
library's tables are created by a *provisioning* role, row-level security is
then enabled and FORCEd on them, and every engine that indexes or retrieves
runs as a restricted grant role. The grant role must not own the tables, and
that is not a preference: an owner may `ALTER TABLE ... NO FORCE ROW LEVEL
SECURITY`, which removes the wall RLS was put up to be.

The lexical channel broke it. `_ensure_relation_fts_index` ran `CREATE INDEX IF
NOT EXISTS` on the first lexical query, and PostgreSQL requires ownership of
the table for CREATE INDEX -- checking for it *before* it evaluates IF NOT
EXISTS. So the statement was refused to the grant role even when the index
already existed, which means no amount of preparation by the deployer could
make the lazy path safe. Only not issuing the statement can, which is what the
catalogue check does.

The second test below is the one that decides the design: it measures that the
"create it in advance as the owner" workaround does not work, so catching the
error and continuing is not equivalent to looking first.
"""

import uuid

import asyncpg
import pytest
from conftest import TEST_DSN, FakeLLM, _db_reachable, make_config

from post_graph_rag import GraphRAG
from post_graph_rag.errors import SchemaError


async def _grant_realm_to_role(admin, realm, role):
    """Rows but not ownership: the grant role's whole reach.

    Includes USAGE on the schema holding the `vector` extension, without which
    every vector read fails with `type "vector" does not exist`.
    """
    ext_schema = await admin.fetchval(
        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
        "ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
    if ext_schema:
        await admin.execute(f'GRANT USAGE ON SCHEMA "{ext_schema}" TO "{role}"')
    await admin.execute(f'GRANT USAGE ON SCHEMA "{realm}" TO "{role}"')
    await admin.execute(
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{realm}" '
        f'TO "{role}"')
    await admin.execute(
        f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "{realm}" TO "{role}"')


def _role_pool(role):
    """A pool that assumes *role* on every acquisition, session-level."""
    async def setup(conn):
        await conn.execute(f'SET ROLE "{role}"')
    return asyncpg.create_pool(TEST_DSN, setup=setup, min_size=1, max_size=4)


@pytest.fixture
def role_name():
    return f"grant_{uuid.uuid4().hex[:10]}"


async def _provisioned_realm(**cfg):
    """A realm initialised and indexed by the owning credential."""
    config = make_config(**cfg)
    if not await _db_reachable(config):
        pytest.skip("PostgreSQL with pgvector not reachable")
    rag = GraphRAG(config)
    rag.llm = FakeLLM(config, answer="provisioned")
    rag.extractor.llm_service = rag.llm
    await rag.initialize()
    await rag.index_document(
        "Vistra Corp reported refining operating costs of 412 million dollars "
        "under statute section 737-9.",
        metadata={"document": "transcript"},
    )
    return rag, config


class TestTheIndexIsThereBeforeTheFirstQuery:
    """Option one: a deployment that provisions as the owner has every index."""

    @pytest.mark.asyncio
    async def test_initialize_schema_creates_the_lexical_index(self):
        rag, config = await _provisioned_realm(lexical_search=True)
        try:
            present = await rag.store._relation_exists(f"{config.realm}_relations_fts")
            assert present, "initialize_schema must leave the lexical index in place"
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await rag.close()

    @pytest.mark.asyncio
    async def test_it_is_not_created_when_the_channel_is_off(self):
        """The reason laziness existed: a realm that never queries lexically
        should not pay for a GIN index on every relation write."""
        rag, config = await _provisioned_realm(lexical_search=False)
        try:
            present = await rag.store._relation_exists(f"{config.realm}_relations_fts")
            assert not present
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await rag.close()


class TestWhyLookingFirstIsNotOptional:
    """The measurement that rules out the alternatives."""

    @pytest.mark.asyncio
    async def test_create_index_if_not_exists_is_refused_to_a_non_owner_anyway(
        self, role_name
    ):
        """`IF NOT EXISTS` does not save a non-owner, and this is the whole point.

        PostgreSQL checks ownership of the table before it looks for the
        index, so the statement raises 42501 even though it would have done
        nothing. That is why the deployer cannot fix this by creating the
        index in advance, and why the library has to ask the catalogue rather
        than try the statement and recover.
        """
        rag, config = await _provisioned_realm(lexical_search=True)
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await rag.close()
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)

            index = f"{config.realm}_relations_fts"
            # It exists -- initialize_schema made it.
            assert await admin.fetchval(
                "SELECT to_regclass($1) IS NOT NULL", f'"{config.realm}"."{index}"')

            conn = await asyncpg.connect(TEST_DSN)
            try:
                await conn.execute(f'SET ROLE "{role_name}"')
                with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc:
                    await conn.execute(
                        f'CREATE INDEX IF NOT EXISTS "{index}" '
                        f'ON "{config.realm}"."relations" USING gin ('
                        f"to_tsvector('english', coalesce(relation_type, '')))")
                assert "must be owner" in str(exc.value).lower()

                # ...while the catalogue read the library now does instead is
                # permitted to the same role on the same connection.
                assert await conn.fetchval(
                    "SELECT to_regclass($1) IS NOT NULL",
                    f'"{config.realm}"."{index}"') is True
            finally:
                await conn.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            try:
                await admin.execute(f'DROP OWNED BY "{role_name}"')
                await admin.execute(f'DROP ROLE "{role_name}"')
            except Exception:
                pass
            await admin.close()

    @pytest.mark.asyncio
    async def test_the_query_path_emits_no_ddl_when_the_index_exists(self):
        """Option two, asserted on the emitted SQL rather than on its effect."""
        rag, config = await _provisioned_realm(lexical_search=True)
        try:
            rag.store._fts_index_ready = False          # as a fresh process sees it
            statements = []
            real_execute = rag.store.client._execute

            async def spy(sql, *args):
                statements.append(" ".join(str(sql).split()))
                return await real_execute(sql, *args)

            rag.store.client._execute = spy
            try:
                await rag.store._ensure_relation_fts_index()
            finally:
                rag.store.client._execute = real_execute

            assert rag.store._fts_index_ready is True
            ddl = [s for s in statements if "CREATE" in s.upper()]
            assert not ddl, f"the query path issued DDL: {ddl}"
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await rag.close()


class TestRetrievalUnderTheGrantRole:
    """The reported failure, end to end."""

    @pytest.mark.asyncio
    async def test_the_lexical_channel_works_for_a_non_owner(self, role_name):
        rag, config = await _provisioned_realm(lexical_search=True)
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await rag.close()
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)

            pool = await _role_pool(role_name)
            try:
                seat = GraphRAG(config, connection=pool)
                seat.llm = FakeLLM(config, answer="from the case schema")
                seat.extractor.llm_service = seat.llm
                async with pool.acquire() as c:
                    assert await c.fetchval("SELECT current_user") == role_name

                # This is the call that raised `must be owner of table relations`.
                hits = await seat.store.search_relations_text("refining costs 737-9")
                assert isinstance(hits, list)
                # And the channel really ran: the fts index is marked ready
                # without the store having created anything.
                assert seat.store._fts_index_ready is True
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
    async def test_a_realm_without_the_index_says_what_to_do(self, role_name):
        """The residual case, and the error it now gets.

        A realm provisioned before this release, or with the channel off, has
        no index -- and a non-owner cannot build one. asyncpg's `must be owner
        of table relations` names a table the caller never asked to alter and
        no remedy at all; this names both.
        """
        rag, config = await _provisioned_realm(lexical_search=False)
        admin = await asyncpg.connect(TEST_DSN)
        try:
            assert not await rag.store._relation_exists(
                f"{config.realm}_relations_fts")
            await rag.close()
            await admin.execute(f'CREATE ROLE "{role_name}" NOLOGIN')
            await _grant_realm_to_role(admin, config.realm, role_name)

            pool = await _role_pool(role_name)
            try:
                seat = GraphRAG(config, connection=pool)
                seat.llm = FakeLLM(config)
                seat.extractor.llm_service = seat.llm
                with pytest.raises(SchemaError) as exc:
                    await seat.store._ensure_relation_fts_index()
                message = str(exc.value)
                assert "initialize_schema" in message
                assert "lexical_search" in message
                assert f"{config.realm}_relations_fts" in message
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


class TestTheTelemetryTableHasTheSameShape:
    """`record_retrieval_event` creates its table lazily too, and swallows
    failures -- so a non-owner's refusal there is silent, which is worse."""

    @pytest.mark.asyncio
    async def test_no_ddl_when_the_events_table_already_exists(self):
        rag, config = await _provisioned_realm(record_retrieval_events=True)
        try:
            assert await rag.store._relation_exists("retrieval_events")
            rag.store._events_table_ready = False       # as a fresh process sees it

            statements = []
            real_execute = rag.store.client._execute

            async def spy(sql, *args):
                statements.append(" ".join(str(sql).split()))
                return await real_execute(sql, *args)

            rag.store.client._execute = spy
            try:
                await rag.store.record_retrieval_event(
                    "mix", "a question", entity_ids=[], community_ids=[])
            finally:
                rag.store.client._execute = real_execute

            ddl = [s for s in statements
                   if "CREATE TABLE" in s.upper() or "CREATE INDEX" in s.upper()]
            assert not ddl, f"the query path issued DDL: {ddl}"
        finally:
            await rag.store.client._execute(
                f'DROP SCHEMA IF EXISTS "{config.realm}" CASCADE')
            await rag.close()
