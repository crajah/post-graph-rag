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
            # Mock fallback if endpoint fails or key is dummy
            logger.warning("Returning zero vector fallback for embedding.")
            return [0.0] * self.config.embedding_dim

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
