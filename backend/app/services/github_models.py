"""
GitHub Models Service - OpenAI SDK Integration

Uses GitHub Marketplace Models via the OpenAI-compatible API.
Includes conversation history support and error handling with retries.
"""
from openai import OpenAI
from app.core.config import settings
from typing import List, Generator, Optional
import time
import logging

logger = logging.getLogger(__name__)


class GitHubModelsService:
    """Service for interacting with GitHub Models (OpenAI-compatible API)."""
    
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.GITHUB_MODELS_ENDPOINT,
            api_key=settings.GITHUB_TOKEN
        )
        self.chat_model = settings.CHAT_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        self._max_retries = 3
    
    def chat_completion(
        self, 
        messages: List[dict], 
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a chat completion response with retry logic."""
        for attempt in range(self._max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Chat completion attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Chat completion failed after {self._max_retries} attempts")
                    raise
    
    def chat_completion_stream(
        self, 
        messages: List[dict], 
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Generate a streaming chat completion response."""
        try:
            stream = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            yield f"\n\n⚠️ Error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with retry."""
        for attempt in range(self._max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.warning(f"Embeddings attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.get_embeddings([text])[0]


# Singleton instance
github_models = GitHubModelsService()
