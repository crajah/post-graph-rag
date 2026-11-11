"""Pool options the caller must be able to set, and who owns what gets created.

Two things a deployment behind PgBouncer could not express through RAGConfig,
each of which forced it to reach past this library and build its own
AsyncPostGraph:

  * `statement_cache_size`. In transaction pooling mode server connections
    rotate between transactions, so asyncpg's per-connection prepared
    statements collide under concurrency as DuplicatePreparedStatementError.
    0 turns them off; the config had no way to say so.
  * `server_settings`, for a session `statement_timeout` where the pooler
    discards the one in the startup packet.

And one that cost a production outage: a batch writer role that is the first
session to reach a realm creates its tables, and owns them. The owning role
can then neither ALTER them nor let pg_dump take ACCESS SHARE, so nightly
backups fail on tables it is supposed to own. `owner_role` closes that at the
end of initialize_schema.
"""

import inspect
import re
import uuid

import asyncpg
import pytest
from conftest import TEST_DSN, FakeLLM, _db_reachable, make_config

import post_graph_rag.config
from post_graph_rag import GraphRAG, RAGConfig
from post_graph_rag.graph_store import RAGGraphStore


class TestPoolOptionsReachAsyncpg:
    """No database needed: the options are on the client before connect()."""

    def test_both_options_are_forwarded(self):
        store = RAGGraphStore(make_config(
            statement_cache_size=0,
            server_settings={"statement_timeout": "15000"}))
        assert store.client.conn_kwargs["statement_cache_size"] == 0
        assert store.client.conn_kwargs["server_settings"] == {
            "statement_timeout": "15000"}

    def test_unset_options_are_not_forwarded(self, monkeypatch):
        """Existing callers see no change: asyncpg keeps its own defaults
        rather than acquiring this module's idea of them."""
        monkeypatch.delenv("RAG_STATEMENT_CACHE_SIZE", raising=False)
        store = RAGGraphStore(make_config())
        assert "statement_cache_size" not in store.client.conn_kwargs
        assert "server_settings" not in store.client.conn_kwargs

    def test_server_settings_are_copied_not_aliased(self):
        """A caller mutating its dict afterwards must not change the pool."""
        settings = {"statement_timeout": "15000"}
        store = RAGGraphStore(make_config(server_settings=settings))
        settings["statement_timeout"] = "1"
        assert store.client.conn_kwargs["server_settings"] == {
            "statement_timeout": "15000"}

    def test_a_supplied_connection_takes_neither(self):
        """They configure a pool this store opens. With one supplied there is
        no pool to configure, and silently ignoring them would be worse than
        not offering them."""
        class _Pool:
            async def acquire(self):      # pragma: no cover - never called
                raise AssertionError

        store = RAGGraphStore(make_config(statement_cache_size=0),
                              connection=_Pool())
        assert store.client.conn_kwargs == {}

    def test_statement_cache_size_reads_its_environment_variable(self, monkeypatch):
        monkeypatch.setenv("RAG_STATEMENT_CACHE_SIZE", "0")
        assert RAGConfig().statement_cache_size == 0
        monkeypatch.setenv("RAG_STATEMENT_CACHE_SIZE", "")
        assert RAGConfig().statement_cache_size is None


class TestTheEnvironmentContract:
    """Invariants a downstream service enforces by scraping this module.

    Asserted here so a change to config.py fails in this repository rather
    than in theirs.
    """

    def test_every_environment_variable_is_namespaced(self):
        names = re.findall(r'_env\(\s*"([A-Z0-9_]+)"',
                           inspect.getsource(post_graph_rag.config))
        assert names
        stray = [n for n in names
                 if not n.startswith(("RAG_", "OPENAI_")) and n != "POSTGRES_URI"]
        assert not stray, stray

    def test_every_variable_parses_when_set_to_one(self, monkeypatch):
        """A config that raises on "1" breaks a downstream masking test, and
        is the reason server_settings is a plain field with no _env behind it:
        a mapping has no sensible reading of a bare string."""
        names = re.findall(r'_env\(\s*"([A-Z0-9_]+)"',
                           inspect.getsource(post_graph_rag.config))
        for name in names:
            if name in ("POSTGRES_URI", "OPENAI_API_BASE"):
                continue
            monkeypatch.setenv(name, "1")
        RAGConfig()        # must not raise


class TestOwnerRole:

    def test_it_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv("RAG_OWNER_ROLE", raising=False)
        assert RAGConfig().owner_role is None

    def test_the_store_exposes_the_pass_through(self):
        assert hasattr(RAGGraphStore, "reassign_owned_to")

    @pytest.mark.asyncio
    async def test_a_hostile_role_is_refused(self):
        store = RAGGraphStore(make_config())
        with pytest.raises(ValueError):
            await store.reassign_owned_to('x"; DROP SCHEMA y; --')


class TestOwnershipAfterInitializeSchema:
    """The outage, reproduced and then prevented."""

    @staticmethod
    def _dsn_as(role):
        head, _, tail = TEST_DSN.partition("://")
        return f"{head}://{role}:{role}@{tail.split('@')[-1]}"

    async def _roles(self, admin, owner, writer, realm):
        for name, extra in ((owner, ""), (writer, f'IN ROLE "{owner}"')):
            await admin.execute(
                f'CREATE ROLE "{name}" LOGIN INHERIT PASSWORD \'{name}\' {extra} '
                f"NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE")
        db = await admin.fetchval("SELECT current_database()")
        await admin.execute(
            f'GRANT CREATE, CONNECT ON DATABASE "{db}" TO "{owner}", "{writer}"')
        await admin.execute(f'GRANT USAGE ON SCHEMA public TO "{owner}", "{writer}"')
        ext = await admin.fetchval(
            "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
            "ON n.oid = e.extnamespace WHERE e.extname = 'vector'")
        if ext:
            await admin.execute(
                f'GRANT USAGE ON SCHEMA "{ext}" TO "{owner}", "{writer}"')
        await admin.execute(f'CREATE SCHEMA "{realm}" AUTHORIZATION "{owner}"')
        await admin.execute(f'GRANT CREATE, USAGE ON SCHEMA "{realm}" TO "{writer}"')

    @pytest.mark.asyncio
    async def test_without_owner_role_the_writer_keeps_what_it_made(self):
        """The control. Without this the test below proves nothing."""
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        tag = uuid.uuid4().hex[:8]
        owner, writer, realm = f"own_{tag}", f"wr_{tag}", config.realm
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await self._roles(admin, owner, writer, realm)
            store = RAGGraphStore(make_config(
                db_uri=self._dsn_as(writer), realm=realm))
            await store.connect()
            try:
                await store.initialize_schema()
            finally:
                await store.close()
            owners = {r["tableowner"] for r in await admin.fetch(
                "SELECT tableowner FROM pg_tables WHERE schemaname = $1", realm)}
            assert owners == {writer}
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                try:
                    await admin.execute(f'DROP OWNED BY "{r}"')
                    await admin.execute(f'DROP ROLE "{r}"')
                except Exception:
                    pass
            await admin.close()

    @pytest.mark.asyncio
    async def test_owner_role_hands_everything_over_before_returning(self):
        config = make_config()
        if not await _db_reachable(config):
            pytest.skip("PostgreSQL with pgvector not reachable")
        tag = uuid.uuid4().hex[:8]
        owner, writer, realm = f"own_{tag}", f"wr_{tag}", config.realm
        admin = await asyncpg.connect(TEST_DSN)
        try:
            await self._roles(admin, owner, writer, realm)
            store = RAGGraphStore(make_config(
                db_uri=self._dsn_as(writer), realm=realm, owner_role=owner))
            await store.connect()
            try:
                await store.initialize_schema()
            finally:
                await store.close()

            tables = {r["tablename"]: r["tableowner"] for r in await admin.fetch(
                "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = $1",
                realm)}
            for expected in ("documents", "documents_data", "documents_audit",
                             "entities", "communities", "relations",
                             "doc_mentions", "community_members",
                             "community_children"):
                assert tables.get(expected) == owner, (expected, tables.get(expected))
            seqs = {r["owner"] for r in await admin.fetch(
                "SELECT pg_get_userbyid(c.relowner) AS owner FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind IN ('S', 'i')", realm)}
            assert seqs == {owner}, seqs

            # And the owner can now use the realm it owns.
            as_owner = RAGGraphStore(make_config(
                db_uri=self._dsn_as(owner), realm=realm))
            await as_owner.connect()
            try:
                await as_owner.initialize_schema()
                assert await as_owner.client.count_vertices(
                    "documents", realm=realm) == 0
            finally:
                await as_owner.close()

            # ...while the writer still writes, inheriting the owner's rights.
            rag = GraphRAG(make_config(db_uri=self._dsn_as(writer), realm=realm,
                                       owner_role=owner))
            rag.llm = FakeLLM(rag.config, answer="x")
            rag.extractor.llm_service = rag.llm
            await rag.initialize()
            try:
                await rag.store.add_document("a passage", [0.1] * config.embedding_dim)
                assert await rag.store.client.count_vertices(
                    "documents", realm=realm) == 1
            finally:
                await rag.close()
        finally:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
            for r in (writer, owner):
                try:
                    await admin.execute(f'DROP OWNED BY "{r}"')
                    await admin.execute(f'DROP ROLE "{r}"')
                except Exception:
                    pass
            await admin.close()
