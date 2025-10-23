"""
Centralized configuration management for the Islamic Chatbot RAG system.

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class for the application."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "islamic_knowledge")
    
    # Ollama Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    OLLAMA_RERANKER_MODEL: Optional[str] = os.getenv("OLLAMA_RERANKER_MODEL", None)
    
    # Embedding Configuration (for sentence-transformers fallback)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-GGUF")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1024"))
    
    # RAG Configuration
    MAX_SOURCES: int = int(os.getenv("MAX_SOURCES", "10"))
    RERANK_WEIGHT: float = float(os.getenv("RERANK_WEIGHT", "0.7"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    CHUNK_SIZE_MAX: int = int(os.getenv("CHUNK_SIZE_MAX", "1500"))
    CHUNK_SIZE_MIN: int = int(os.getenv("CHUNK_SIZE_MIN", "100"))
    
    # Application Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
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

