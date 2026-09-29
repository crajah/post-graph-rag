"""Every replica calls initialize_schema, and they all have to survive it.

This is where the concurrency bug actually shows up for a user: not in a
synthetic DDL race but in a deployment whose replicas all provision the same
realm on the first write to a new space. Measured before the fix, six replicas
produced five SchemaErrors and one success.
"""
import asyncio
import uuid

import pytest
from conftest import make_config

from post_graph_rag import GraphRAG

REPLICAS = 6


async def _race(n, **overrides):
    realm = "ci_" + uuid.uuid4().hex[:10]
    rags = [GraphRAG(make_config(realm=realm, schema_per_realm=True, **overrides))
            for _ in range(n)]
    try:
        outcomes = await asyncio.gather(*[r.initialize() for r in rags],
                                        return_exceptions=True)
        yield rags, outcomes
    finally:
        try:
            await rags[0].store.client._execute(f'DROP SCHEMA IF EXISTS "{realm}" CASCADE')
        except Exception:
            pass
        for r in rags:
            try:
                await r.close()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_replicas_racing_to_initialize_a_realm(rag_factory):
    # Establishes DB reachability and skips cleanly when there is none.
    await rag_factory()

    gen = _race(REPLICAS)
    rags, outcomes = await gen.__anext__()
    try:
        failed = [o for o in outcomes if isinstance(o, BaseException)]
        assert not failed, f"{len(failed)}/{REPLICAS} replicas failed: {failed[:2]}"

        # Provisioned by a stampede, and still usable: the entity name index
        # is what upsert_entity resolves against, and it is the statement that
        # used to fail.
        v = await rags[0].store.upsert_entity(
            "Zeus", "Person", "king of the gods", [0.1] * 16, space="default")
        assert v is not None
        again = await rags[1].store.upsert_entity(
            "Zeus", "Person", "king of the gods", [0.1] * 16, space="default")
        assert again.id == v.id, "the uniqueness index must really be in place"
    finally:
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


@pytest.mark.asyncio
async def test_initializing_twice_from_one_client_is_still_fine(rag_factory):
    """Idempotence, which the race tolerance must not have broken."""
    rag = await rag_factory()
    await rag.store.initialize_schema()
    await rag.store.initialize_schema()
    v = await rag.store.upsert_entity("Hera", "Person", "d", [0.1] * 16, space="default")
    assert v is not None
