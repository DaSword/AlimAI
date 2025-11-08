"""
Centralized configuration management for the Islamic Chatbot RAG system.

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class for the application."""
    
    # Base directories (backend/core/config.py -> alimai/)
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "islamic_knowledge_huggingface")
    
    # Embedding Configuration
    # Backend options: "huggingface" (fast, local), "ollama" (API-based), or "lmstudio" (local server)
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "huggingface")
    
    # LLM Backend Configuration
    # Backend options: "ollama" (default), "lmstudio" (local server)
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "lmstudio")
    
    # HuggingFace/SentenceTransformers model (when EMBEDDING_BACKEND="huggingface")
    # Fast, local embeddings with true batching
    # Options: "google/embeddinggemma-300m", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
    
    # Ollama Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")  # When EMBEDDING_BACKEND="ollama"
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
    OLLAMA_RERANKER_MODEL: Optional[str] = os.getenv("OLLAMA_RERANKER_MODEL", "dengcao/Qwen3-Reranker-0.6B:Q8_0")
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))
    OLLAMA_REQUEST_TIMEOUT: int = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "100"))
    OLLAMA_EMBEDDING_BATCH_SIZE: int = int(os.getenv("OLLAMA_EMBEDDING_BATCH_SIZE", "5"))
    
    # LM Studio Configuration
    LMSTUDIO_URL: str = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
    LMSTUDIO_EMBEDDING_MODEL: str = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m-qat")  # When EMBEDDING_BACKEND="lmstudio"
    # LMSTUDIO_CHAT_MODEL: str = os.getenv("LMSTUDIO_CHAT_MODEL", "")  # When LLM_BACKEND="lmstudio"
    LMSTUDIO_CHAT_MODEL: str = os.getenv("LMSTUDIO_CHAT_MODEL", "qwen/qwen3-vl-8b")  # When LLM_BACKEND="lmstudio"
    LMSTUDIO_RERANKER_MODEL: Optional[str] = os.getenv("LMSTUDIO_RERANKER_MODEL", "hermes-2-pro-llama-3-8b")
    LMSTUDIO_MAX_TOKENS: int = int(os.getenv("LMSTUDIO_MAX_TOKENS", "1000"))
    LMSTUDIO_REQUEST_TIMEOUT: int = int(os.getenv("LMSTUDIO_REQUEST_TIMEOUT", "1000"))
    
    # Vector Configuration
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))  # embeddinggemma default
    
    # RAG Configuration
    MAX_SOURCES: int = int(os.getenv("MAX_SOURCES", "10"))
    RERANK_WEIGHT: float = float(os.getenv("RERANK_WEIGHT", "0.7"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    CHUNK_SIZE_MAX: int = int(os.getenv("CHUNK_SIZE_MAX", "1500"))
    CHUNK_SIZE_MIN: int = int(os.getenv("CHUNK_SIZE_MIN", "100"))
    
    # Application Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Streaming Configuration
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() in ("true", "1", "yes")
    
    # Server Configuration (for future LangGraph Server)
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8123"))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path for a data file."""
        return cls.DATA_DIR / filename


# Create a singleton instance
config = Config()

# Ensure directories exist on import
config.ensure_directories()

