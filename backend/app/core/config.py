from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Workflow Copilot"
    API_V1_STR: str = "/api/v1"
    
    # GitHub Models Configuration
    GITHUB_TOKEN: str = ""
    GITHUB_MODELS_ENDPOINT: str = "https://models.inference.ai.azure.com"
    
    # Model Configuration
    CHAT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # ChromaDB Configuration (persistent)
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    
    # RAG Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    # Hybrid Search
    BM25_WEIGHT: float = 0.3  # Weight for BM25 (semantic gets 1 - this)
    RERANKER_ENABLED: bool = False
    
    # Guardrails
    GUARDRAILS_ENABLED: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = "../.env"


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
