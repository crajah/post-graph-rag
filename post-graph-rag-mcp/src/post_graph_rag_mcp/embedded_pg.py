"""Multi-window safe embedded PostgreSQL lifecycle manager via pgserver."""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
import sys
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

_EMBEDDED_INSTALL_CMD = (
    'uvx --python 3.12 --from "post-graph-rag-mcp[embedded,local-embed]" post-graph-rag-mcp'
)


def _is_uri_reachable(uri: str, timeout: float = 1.5) -> bool:
    """Return True if *uri* resolves to a live TCP or Unix-domain PostgreSQL listener.

    Uses stdlib ``urllib.parse`` and ``socket`` only (no SQL drivers).
    """
    cleaned = (uri or "").strip()
    if not cleaned:
        return False

    try:
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme not in ("postgres", "postgresql"):
            return False

        params = urllib.parse.parse_qs(parsed.query)
        host_param = (params.get("host") or [None])[0]
        port_raw = (params.get("port") or [None])[0]
        port = int(port_raw) if port_raw is not None else (parsed.port or 5432)

        raw_host = urllib.parse.unquote(host_param) if host_param else (parsed.hostname or "")

        if raw_host.startswith("/") or (os.name != "nt" and os.path.isabs(raw_host)):
            if not hasattr(socket, "AF_UNIX"):
                return False
            sock_base = Path(raw_host)
            sock_path = (
                sock_base
                if sock_base.name.startswith(".s.PGSQL.")
                else sock_base / f".s.PGSQL.{port}"
            )
            if not os.path.exists(sock_path):
                return False
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(str(sock_path))
            return True

        tcp_host = raw_host or "127.0.0.1"
        with socket.create_connection((tcp_host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _atomic_write_uri(uri_file: Path, uri: str) -> None:
    uri_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = uri_file.parent / f".mcp_uri.{os.getpid()}.tmp"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{uri}\n")
        os.replace(tmp_path, uri_file)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


class EmbeddedPostgresManager:
    """Manages an embedded PostgreSQL + pgvector instance shared across IDE windows.

    Coordinates startup via ``fcntl.flock`` on ``.<pgdata_name>.lock`` and reuses
    an existing postmaster recorded in ``<pgdata>/.mcp_uri`` whenever its socket
    is reachable, avoiding duplicate postmaster processes and preventing single-window
    shutdowns from wiping shared data.
    """

    def __init__(self, pgdata_dir: Path) -> None:
        self.pgdata_dir = Path(pgdata_dir).expanduser().resolve()
        self._lock_path = self.pgdata_dir.parent / f".{self.pgdata_dir.name}.lock"
        self._uri_file = self.pgdata_dir / ".mcp_uri"
        self._async_lock = asyncio.Lock()
        self._cached_uri: str | None = None
        self._server: Any = None

    def _ensure_uri_sync(self) -> str:
        with _exclusive_file_lock(self._lock_path):
            if self._uri_file.is_file():
                try:
                    recorded_uri = self._uri_file.read_text(encoding="utf-8").strip()
                except OSError:
                    recorded_uri = ""
                if recorded_uri and _is_uri_reachable(recorded_uri):
                    return recorded_uri

            try:
                import pgserver  # type: ignore[import-not-found]
            except ImportError as exc:
                py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
                detail = (
                    f"running on Python {py_ver}, whereas pgserver binary wheels require Python < 3.13"
                    if sys.version_info >= (3, 13)
                    else "the optional 'pgserver' package is not installed"
                )
                raise RuntimeError(
                    f"No POSTGRES_URI is configured and embedded PostgreSQL is unavailable ({detail}). "
                    "Set POSTGRES_URI to connect to an existing PostgreSQL + pgvector database, "
                    "or launch with embedded PostgreSQL on Python 3.12:\n"
                    f"  {_EMBEDDED_INSTALL_CMD}"
                ) from exc

            self.pgdata_dir.parent.mkdir(parents=True, exist_ok=True)

            sig = inspect.signature(pgserver.get_server)
            kwargs: dict[str, Any] = {}
            if "cleanup_mode" in sig.parameters:
                kwargs["cleanup_mode"] = None

            server = pgserver.get_server(str(self.pgdata_dir), **kwargs)
            if not kwargs and hasattr(server, "cleanup_mode"):
                server.cleanup_mode = None

            self._server = server
            uri = str(server.get_uri()).strip()
            if not uri:
                raise RuntimeError(
                    f"pgserver started for '{self.pgdata_dir}' but returned an empty connection URI."
                )

            _atomic_write_uri(self._uri_file, uri)
            return uri

    async def ensure_uri(self) -> str:
        """Return a live PostgreSQL connection URI, booting ``pgserver`` if needed."""
        async with self._async_lock:
            if self._cached_uri and _is_uri_reachable(self._cached_uri):
                return self._cached_uri
            uri = await asyncio.to_thread(self._ensure_uri_sync)
            self._cached_uri = uri
            return uri

    async def close(self) -> None:
        """Release in-process state without stopping or deleting shared ``pgdata``."""
        async with self._async_lock:
            self._cached_uri = None
