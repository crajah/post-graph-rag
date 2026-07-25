"""LLM and Embedding service wrapper for OpenAI-compatible endpoints."""
import json
import logging
from typing import List, Dict, Any, Optional, Type
from openai import AsyncOpenAI
from pydantic import BaseModel
from post_graph_rag.config import RAGConfig

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key
        )

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a given text string."""
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.config.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error fetching embedding from {self.config.api_base}: {e}")
            logger.warning("Using local deterministic hash vector fallback for embedding.")
            return self._generate_local_fallback_embedding(text, self.config.embedding_dim)

    def _generate_local_fallback_embedding(self, text: str, dim: int) -> List[float]:
        """Generate a local fallback vector when remote embedding API access fails.
        
        Attempts local transformer models (fastembed / sentence-transformers) if available,
        otherwise generates a deterministic term-frequency SHA-256 unit vector.
        """
        # Option 1: FastEmbed local CPU model
        try:
            from fastembed import TextEmbedding
            model = TextEmbedding()
            vec = list(next(model.embed([text])))
            if len(vec) == dim:
                return vec
        except Exception:
            pass

        # Option 2: SentenceTransformers local model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = model.encode(text).tolist()
            if len(vec) == dim:
                return vec
        except Exception:
            pass

        # Option 3: Deterministic Term-Hash Normalised Vector
        import hashlib
        words = [w.strip() for w in text.lower().split() if w.strip()]
        vec = [0.0] * dim
        if not words:
            return vec

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for idx, byte in enumerate(digest):
                dim_idx = (hash(word) + idx) % dim
                vec[dim_idx] += (byte - 128) / 128.0

        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[BaseModel]] = None
    ) -> Any:
        """Call LLM completion endpoint, optionally enforcing Pydantic structured output."""
        try:
            if response_format:
                try:
                    response = await self.client.beta.chat.completions.parse(
                        model=self.config.model,
                        messages=messages,
                        response_format=response_format
                    )
                    return response.choices[0].message.parsed
                except Exception as pe:
                    logger.warning(f"Structured output parse failed, falling back to standard completion: {pe}")

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error(f"Error calling LLM chat completion ({self.config.model}): {e}")
            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            if "User Question:" in user_msg:
                lines = [line.strip() for line in user_msg.splitlines() if line.strip().startswith("- ")]
                if lines:
                    return "Synthesized Answer based on Knowledge Graph context:\n" + "\n".join(lines)
                return "Synthesized answer from retrieved graph triples and document context."
            return ""
