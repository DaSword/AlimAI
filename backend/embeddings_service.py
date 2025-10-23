"""
Embeddings Service - Generate text embeddings for Islamic texts.

This module provides a wrapper around sentence-transformers for backward compatibility
and will be updated to use Ollama via LlamaIndex in Stage 3.

Migrated from embeddings_manager.py with updated imports for backend structure.
"""

import time
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

# Updated imports for backend structure
from backend.config import Config
from backend.utils import setup_logging

logger = setup_logging("embeddings_service")


class EmbeddingsService:
    """
    Service class for generating text embeddings.
    
    Currently uses sentence-transformers. Will be updated in Stage 3 to use
    Ollama embeddings via LlamaIndex (OllamaEmbedding).
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize EmbeddingsService.
        
        Args:
            model_name: Embedding model name (defaults to config)
        """
        # Use SentenceTransformer with Qwen3 model from HuggingFace
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.embedding_model = None  # Will be initialized in check_model_availability
        self.vector_size = None  # Will be detected from model
        
        logger.info(f"Initialized EmbeddingsService with model: {self.model_name}")
        
    def _load_model(self) -> bool:
        """
        Load the embedding model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Detect vector dimensions
            test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
            self.vector_size = len(test_embedding)
            
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"📐 Vector dimensions: {self.vector_size}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {str(e)}")
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if the embedding model can be loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        if self.embedding_model is None:
            return self._load_model()
        return True
    
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
        
        # Ensure model is loaded
        if self.embedding_model is None:
            if not self._load_model():
                return None
        
        try:
            # Generate embedding using SentenceTransformer
            embedding = self.embedding_model.encode(
                text.strip(),
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_delay: float = 0.1,
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently using batch encoding.
        
        Args:
            texts: List of texts to embed
            batch_delay: Delay between batches (unused, kept for compatibility)
            show_progress: Whether to show progress updates
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        # Ensure model is loaded
        if self.embedding_model is None:
            if not self._load_model():
                return [None] * len(texts)
        
        total = len(texts)
        
        if show_progress:
            logger.info(f"Generating embeddings for {total} texts...")
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            # Convert numpy arrays to lists for JSON serialization
            return [embedding.tolist() for embedding in embeddings]
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
                "expected_vector_size": self.vector_size,
                "generation_time": round(elapsed_time, 3),
                "sample_values": embedding[:5]  # First 5 values
            }
            
            logger.info("✓ Test successful!")
            logger.info(f"  Vector size: {len(embedding)}")
            logger.info(f"  Generation time: {elapsed_time:.3f}s")
            logger.info(f"  Sample values: {embedding[:5]}")
        else:
            result = {
                "success": False,
                "model": self.model_name,
                "error": "Failed to generate embedding"
            }
            logger.error("✗ Test failed!")
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the embeddings configuration.
        
        Returns:
            Dictionary with configuration info
        """
        return {
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "model_loaded": self.embedding_model is not None
        }
    
    def print_info(self):
        """Print embeddings configuration in a readable format."""
        info = self.get_info()
        print(f"\n{'='*60}")
        print(f"Embeddings Service Configuration")
        print(f"{'='*60}")
        print(f"Model: {info['model_name']}")
        print(f"Vector Size: {info['vector_size']}")
        print(f"Model Loaded: {info['model_loaded']}")
        print(f"{'='*60}\n")

    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded")
        
        return self.embedding_model.similarity(vector1, vector2)


# Backward compatibility alias
EmbeddingsManager = EmbeddingsService


def main():
    """Main function to test the embeddings service."""
    # Initialize service
    service = EmbeddingsService()
    
    # Print configuration
    service.print_info()
    
    # Check model availability
    service.check_model_availability()
    
    # Run test
    service.test_embedding()

    # Test cosine similarity
    vector1 = service.generate_embedding("Bismillah ar-Rahman ar-Rahim")
    vector2 = service.generate_embedding("بسم الله الرحمن الرحيم")
    similarity = service.cosine_similarity(vector1, vector2)
    print(f"Cosine similarity: {similarity}")


if __name__ == "__main__":
    main()

