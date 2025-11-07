"""
Embeddings Service - Generate text embeddings for Islamic texts.

This module provides a wrapper around LlamaIndex's HuggingFaceEmbedding for generating
embeddings via SentenceTransformers (much faster than Ollama API).

Updated to use HuggingFace/SentenceTransformers for performance.
"""

import time
from typing import List, Dict, Any, Optional

from backend.core.config import config
from backend.core.utils import setup_logging

logger = setup_logging("embeddings_service")


class EmbeddingsService:
    """
    Service class for generating text embeddings.
    
    Supports multiple backends:
    - Llama.cpp (OpenAI-compatible API)
    - HuggingFace/SentenceTransformers (fast, local, batched)
    - Ollama API (flexible, model variety)
    """
    
    def __init__(
        self,
        embedding_backend: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize EmbeddingsService.
        
        Args:
            embedding_backend: Embedding backend ('llamacpp', 'huggingface' or 'ollama', defaults to config)
            model_name: Embedding model name (defaults to config based on backend)
            device: Device for HuggingFace embeddings ('cuda' or 'cpu', auto-detects if None)
        """
        self.embedding_backend = embedding_backend or config.EMBEDDING_BACKEND
        self.model_name = model_name
        self.device = device
        self.embedding_model = None
        self.vector_size = None
        
        logger.info(f"Initialized EmbeddingsService with backend: {self.embedding_backend}")
        
        self._load_embedding_model()

    def _load_embedding_model(self) -> bool:
        """
        Load the embedding model based on configured backend.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.embedding_backend == "llamacpp":
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.embeddings.openai.base import OpenAIEmbeddingMode
                
                self.model_name = self.model_name or config.LLAMACPP_EMBEDDING_MODEL
                logger.info(f"Loading Llama.cpp embedding model: {self.model_name}")
                logger.info(f"Llama.cpp URL: {config.LLAMACPP_EMBEDDING_URL}")
                
                # For Llama.cpp with custom models, use SIMILARITY_MODE
                self.embedding_model = OpenAIEmbedding(
                    model_name=self.model_name,
                    api_base=config.LLAMACPP_EMBEDDING_URL,
                    api_key="llama-cpp",  # Llama.cpp doesn't require a real API key
                    mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
                )
                
                # Test embedding to detect vector dimensions
                test_embedding = self.embedding_model.get_text_embedding("test")
                self.vector_size = len(test_embedding)
                
                logger.info("✓ Llama.cpp embedding model loaded successfully")
                logger.info(f"✓ Vector dimensions: {self.vector_size}")
                return True
            
            elif self.embedding_backend == "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                
                self.model_name = self.model_name or config.EMBEDDING_MODEL
                logger.info(f"Loading HuggingFace embedding model: {self.model_name}")
                
                # Auto-detect device if not specified
                if self.device is None:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                logger.info(f"Using device: {self.device}")
                
                # Get HuggingFace token from environment if available
                import os
                hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
                
                embed_model_kwargs = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "trust_remote_code": True,
                }
                
                # Add token if available (required for gated models like embeddinggemma)
                if hf_token:
                    logger.info("✓ Using HuggingFace token for authentication")
                    embed_model_kwargs["token"] = hf_token
                
                self.embedding_model = HuggingFaceEmbedding(**embed_model_kwargs)
                
                # Test embedding to detect vector dimensions
                test_embedding = self.embedding_model.get_text_embedding("test")
                self.vector_size = len(test_embedding)
                
                logger.info("✓ HuggingFace embedding model loaded successfully")
                logger.info(f"✓ Vector dimensions: {self.vector_size}")
                return True
                
            elif self.embedding_backend == "ollama":
                from llama_index.embeddings.ollama import OllamaEmbedding
                
                self.model_name = self.model_name or config.OLLAMA_EMBEDDING_MODEL
                logger.info(f"Loading Ollama embedding model: {self.model_name}")
                
                self.embedding_model = OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=config.OLLAMA_URL,
                    request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
                )
                
                # Test embedding to detect vector dimensions
                test_embedding = self.embedding_model.get_text_embedding("test")
                self.vector_size = len(test_embedding)
                
                logger.info("✓ Ollama embedding model loaded successfully")
                logger.info(f"✓ Vector dimensions: {self.vector_size}")
                return True
            
            elif self.embedding_backend == "lmstudio":
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.embeddings.openai.base import OpenAIEmbeddingMode
                
                self.model_name = self.model_name or config.LMSTUDIO_EMBEDDING_MODEL
                logger.info(f"Loading LM Studio embedding model: {self.model_name}")
                logger.info(f"LM Studio URL: {config.LMSTUDIO_URL}")
                
                # For LM Studio with custom models, use SIMILARITY_MODE
                self.embedding_model = OpenAIEmbedding(
                    model_name=self.model_name,
                    api_base=config.LMSTUDIO_URL,
                    api_key="lm-studio",  # LM Studio doesn't require a real API key
                    mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
                )
                
                # Test embedding to detect vector dimensions
                test_embedding = self.embedding_model.get_text_embedding("test")
                self.vector_size = len(test_embedding)
                
                logger.info("✓ LM Studio embedding model loaded successfully")
                logger.info(f"✓ Vector dimensions: {self.vector_size}")
                return True
            
            else:
                logger.error(f"Unknown embedding backend: {self.embedding_backend}")
                logger.error("  Use 'llamacpp', 'huggingface', 'ollama', or 'lmstudio'")
                return False
            
        except ImportError as e:
            logger.error(f"Required package not installed: {str(e)}")
            if self.embedding_backend == "llamacpp":
                logger.error("  Install with: pip install llama-index-embeddings-openai")
            elif self.embedding_backend == "huggingface":
                logger.error("  Install with: pip install llama-index-embeddings-huggingface sentence-transformers")
            else:
                logger.error("  Install with: pip install llama-index-embeddings-ollama")
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            logger.error(f"  Backend: {self.embedding_backend}")
            logger.error(f"  Model: '{self.model_name}'")
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if an embedding model can be loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.embedding_model is not None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        # Ensure a model is loaded
        if not self.check_model_availability():
            logger.error("No embedding model available")
            return None
        
        try:
            embedding = self.embedding_model.get_text_embedding(text.strip())
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently using true batching.
        
        This is much faster than the Ollama API approach as it processes
        texts in parallel using tensor operations.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress updates
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        # Ensure a model is loaded
        if not self.check_model_availability():
            logger.error("No embedding model available")
            return [None] * len(texts)
        
        total = len(texts)
        
        if show_progress:
            logger.info(f"Generating embeddings for {total} texts...")
        
        try:
            # HuggingFaceEmbedding supports true batching - much faster!
            # Clean texts
            clean_texts = [text.strip() for text in texts if text and text.strip()]
            
            # Use the embed_documents method for batch processing
            embeddings = self.embedding_model.get_text_embedding_batch(clean_texts)
            
            if show_progress:
                logger.info(f"✓ Generated {len(embeddings)} embeddings")
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [None] * total
    
    def test_embedding(self) -> Dict[str, Any]:
        """
        Test the embedding generation with a sample text.
        
        Returns:
            Dictionary with test results
        """
        test_text = "Bismillah ar-Rahman ar-Rahim بسم الله الرحمن الرحيم"
        
        logger.info("Testing embedding generation...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Test text: {test_text}")
        
        start_time = time.time()
        embedding = self.generate_embedding(test_text)
        elapsed_time = time.time() - start_time
        
        if embedding:
            result = {
                "success": True,
                "model": self.model_name,
                "vector_size": len(embedding),
                "generation_time": round(elapsed_time, 3),
                "sample_values": embedding[:5]  # First 5 values
            }
            
            logger.info("Test successful!")
            logger.info(f"  Vector size: {len(embedding)}")
            logger.info(f"  Generation time: {elapsed_time:.3f}s")
            logger.info(f"  Sample values: {embedding[:5]}")
        else:
            result = {
                "success": False,
                "model": self.model_name,
                "error": "Failed to generate embedding"
            }
            logger.error("Test failed!")
        
        return result


# Backward compatibility alias
EmbeddingsManager = EmbeddingsService


def main():
    """Main function to test the embeddings service."""
    # Initialize service
    service = EmbeddingsService()
    
    # Check model availability
    if not service.check_model_availability():
        print("Failed to load any embedding model")
        return
    
    # Run test
    result = service.test_embedding()
    print(f"Test result: {result}")
    
    # Test batch embeddings
    print("Testing batch embeddings...")
    texts = [
        "Bismillah ar-Rahman ar-Rahim",
        "بسم الله الرحمن الرحيم",
        "In the name of Allah, the Most Gracious, the Most Merciful"
    ]
    embeddings = service.generate_embeddings_batch(texts, show_progress=True)
    print(f"Generated {len([e for e in embeddings if e])} embeddings successfully")


if __name__ == "__main__":
    main()
