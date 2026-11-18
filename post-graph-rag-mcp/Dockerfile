# syntax=docker/dockerfile:1
# Supports building from repository root:
#   docker build -f post-graph-rag-mcp/Dockerfile -t post-graph-rag-mcp .
# OR directly from the post-graph-rag-mcp/ subdirectory:
#   docker build -t post-graph-rag-mcp post-graph-rag-mcp

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv

COPY . /tmp/build-ctx/

RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    if [ -d /tmp/build-ctx/post-graph-rag-mcp ]; then \
        /opt/venv/bin/pip install --no-cache-dir /tmp/build-ctx "/tmp/build-ctx/post-graph-rag-mcp[local-embed]"; \
    else \
        /opt/venv/bin/pip install --no-cache-dir "/tmp/build-ctx[local-embed]"; \
    fi && \
    rm -rf /tmp/build-ctx


FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="post-graph-rag-mcp" \
      org.opencontainers.image.description="Bi-temporal Graph RAG MCP memory server on PostgreSQL + pgvector" \
      org.opencontainers.image.url="https://github.com/crajah/post-graph-rag" \
      org.opencontainers.image.source="https://github.com/crajah/post-graph-rag" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN groupadd --system --gid 10001 mcpuser && \
    useradd --system --uid 10001 --gid mcpuser --create-home --home-dir /home/mcpuser mcpuser && \
    mkdir -p /data/workspace /home/mcpuser/.cache/fastembed && \
    chown -R mcpuser:mcpuser /data/workspace /home/mcpuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH="/home/mcpuser/.cache/fastembed" \
    MCP_ALLOWED_ROOTS="/data/workspace" \
    MCP_TRANSPORT="streamable-http" \
    MCP_HOST="0.0.0.0" \
    MCP_PORT="8080"

WORKDIR /data/workspace

USER mcpuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, socket; os.getenv('MCP_TRANSPORT', 'streamable-http') != 'streamable-http' or socket.create_connection(('127.0.0.1', int(os.getenv('MCP_PORT', '8080'))), timeout=3).close()"]

ENTRYPOINT ["post-graph-rag-mcp"]
