"""Live-PostgreSQL tests for ADR directory ingestion ordering, idempotency, and path sandboxing."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import pytest


async def test_adr_directory_sync_filename_ordering_supersession_and_unchanged_skip(
    mcp_harness_factory: Callable[..., Any],
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    adr_dir = workspace_root / "adr"
    adr_dir.mkdir(parents=True)

    adr_0001 = adr_dir / "0001-clerk.md"
    adr_0002 = adr_dir / "0002-betterauth.md"

    adr_0001.write_text(
        "---\n"
        "title: ADR 0001 - Adopt Clerk for Authentication\n"
        "entities:\n"
        "  - name: AuthService\n"
        "    type: Technology\n"
        "    description: Core authentication service\n"
        "  - name: Clerk\n"
        "    type: Technology\n"
        "    description: Hosted authentication provider\n"
        "triples:\n"
        "  - subject: AuthService\n"
        "    predicate: uses_auth\n"
        "    object: Clerk\n"
        "---\n\n"
        "# ADR 0001: Adopt Clerk\n\n"
        "AuthService delegates user login and session management to Clerk.\n",
        encoding="utf-8",
    )

    adr_0002.write_text(
        "---\n"
        "title: ADR 0002 - Migrate AuthService to BetterAuth\n"
        "entities:\n"
        "  - name: BetterAuth\n"
        "    type: Technology\n"
        "    description: Self-hosted TypeScript authentication framework\n"
        "triples:\n"
        "  - subject: AuthService\n"
        "    predicate: uses_auth\n"
        "    object: BetterAuth\n"
        "---\n\n"
        "# ADR 0002: Migrate to BetterAuth\n\n"
        "AuthService replaces Clerk with BetterAuth for self-hosted session control.\n",
        encoding="utf-8",
    )

    os.utime(adr_0002, (1_700_000_100, 1_700_000_100))
    os.utime(adr_0001, (1_700_000_900, 1_700_000_900))

    harness = mcp_harness_factory(allowed_roots=[workspace_root])

    first_sync = await harness.call(
        "ingest_documents",
        {
            "paths": [str(adr_dir)],
            "glob": "**/*.md",
            "order_by": "filename",
            "collection": "architecture_decisions",
        },
    )
    assert first_sync["files_scanned"] == 2
    assert first_sync["chunks_indexed"] >= 2
    assert first_sync["chunks_skipped_unchanged"] == 0
    assert first_sync["extraction_mode"] == "chunks_and_frontmatter"
    superseded_total = first_sync.get(
        "superseded_count", first_sync.get("relations_superseded", 0)
    )
    assert superseded_total >= 1

    recalled = await harness.call(
        "recall",
        {
            "question": "Which authentication framework does AuthService use?",
            "entity_keywords": ["AuthService"],
        },
    )
    active_auth = [
        harness.object(r)
        for r in recalled["relationships"]
        if harness.subject(r) == "AuthService"
    ]
    assert active_auth == ["BetterAuth"]

    second_sync = await harness.call(
        "ingest_documents",
        {
            "paths": [str(adr_dir)],
            "glob": "**/*.md",
            "order_by": "filename",
            "collection": "architecture_decisions",
        },
    )
    assert second_sync["files_scanned"] == 2
    assert second_sync["chunks_skipped_unchanged"] >= 2
    assert second_sync["chunks_indexed"] == 0


async def test_ingest_documents_mtime_ordering(
    mcp_harness_factory: Callable[..., Any],
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "mtime_workspace"
    notes_dir = workspace_root / "decisions"
    notes_dir.mkdir(parents=True)

    older_file = notes_dir / "z-initial.md"
    newer_file = notes_dir / "a-migration.md"

    older_file.write_text(
        "---\n"
        "triples:\n"
        "  - subject: QueueWorker\n"
        "    predicate: uses_queue\n"
        "    object: RabbitMQ\n"
        "---\n\n"
        "QueueWorker dispatches background tasks over RabbitMQ.\n",
        encoding="utf-8",
    )
    newer_file.write_text(
        "---\n"
        "triples:\n"
        "  - subject: QueueWorker\n"
        "    predicate: uses_queue\n"
        "    object: BullMQ\n"
        "---\n\n"
        "QueueWorker migrated task queues from RabbitMQ to BullMQ.\n",
        encoding="utf-8",
    )

    os.utime(older_file, (1_700_000_100, 1_700_000_100))
    os.utime(newer_file, (1_700_000_500, 1_700_000_500))

    harness = mcp_harness_factory(allowed_roots=[workspace_root])
    await harness.call(
        "ingest_documents",
        {
            "paths": [str(notes_dir)],
            "glob": "**/*.md",
            "order_by": "mtime",
        },
    )

    recalled = await harness.call(
        "recall",
        {
            "question": "Which queue engine does QueueWorker use?",
            "entity_keywords": ["QueueWorker"],
        },
    )
    active_queues = [
        harness.object(r)
        for r in recalled["relationships"]
        if harness.subject(r) == "QueueWorker"
    ]
    assert active_queues == ["BullMQ"]


async def test_path_traversal_and_symlink_escape_refused_with_permission_error(
    mcp_harness_factory: Callable[..., Any],
    tmp_path: Path,
) -> None:
    allowed_dir = tmp_path / "sandboxed_repo"
    allowed_dir.mkdir(parents=True)

    outside_dir = tmp_path / "outside_secrets"
    outside_dir.mkdir(parents=True)
    secret_doc = outside_dir / "credentials.md"
    secret_doc.write_text("# Sensitive outside file\n", encoding="utf-8")

    escape_link = allowed_dir / "escaped_link.md"
    escape_link.symlink_to(secret_doc)

    harness = mcp_harness_factory(allowed_roots=[allowed_dir])

    with pytest.raises(PermissionError):
        harness.config.validate_ingest_path("../../etc/passwd")

    with pytest.raises(PermissionError):
        await harness.call(
            "ingest_documents",
            {"paths": ["../../etc/passwd"]},
        )

    with pytest.raises(PermissionError):
        await harness.call(
            "ingest_documents",
            {"paths": [str(allowed_dir / ".." / "outside_secrets" / "credentials.md")]},
        )

    with pytest.raises(PermissionError):
        await harness.call(
            "ingest_documents",
            {"paths": [str(escape_link)]},
        )
