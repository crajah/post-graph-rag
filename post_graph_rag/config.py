"""Configuration dataclass for post-graph-rag."""
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    api_base: str = os.getenv("OPENAI_API_BASE", "http://localhost:4000/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "BEVZ-6L81-OZ8Y")
    model: str = os.getenv("RAG_MODEL", "DeepSeek-V3.2")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "E5-Mistral-7B-Instruct")
    embedding_dim: int = int(os.getenv("RAG_EMBEDDING_DIM", "4096"))
    db_uri: str = os.getenv("POSTGRES_URI", "postgresql://crajah@localhost:5432/postgres")
    realm: str = os.getenv("RAG_REALM", "default")
    space: str = os.getenv("RAG_SPACE", "default")
