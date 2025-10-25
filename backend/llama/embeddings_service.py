"""
Embeddings Service - Generate text embeddings for Islamic texts.

This module provides a wrapper around LlamaIndex's OllamaEmbedding for generating
embeddings via Ollama

Updated in Stage 3 to use Ollama via LlamaIndex.
"""

import time
from typing import List, Dict, Any, Optional

from backend.core.config import config
from backend.core.utils import setup_logging

logger = setup_logging("embeddings_service")


class EmbeddingsService:
    """
    Service class for generating text embeddings using Ollama via LlamaIndex.
    
    Uses LlamaIndex's OllamaEmbedding for primary embedding generation.

    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        """
        Initialize EmbeddingsService.
        
        Args:
            model_name: Embedding model name (defaults to config)
        """
        self.model_name = model_name or config.OLLAMA_EMBEDDING_MODEL
        self.ollama_embedding = None
        self.vector_size = None
        
        logger.info(f"Initialized EmbeddingsService")
        
        self._load_ollama_embedding()

    def _load_ollama_embedding(self) -> bool:
        """
        Load the Ollama embedding model via LlamaIndex.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            logger.info(f"Loading Ollama embedding model: {self.model_name}")
            
            self.ollama_embedding = OllamaEmbedding(
                model_name=self.model_name,
                base_url=config.OLLAMA_URL,
                request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
            )
            
            # Test embedding to detect vector dimensions
            test_embedding = self.ollama_embedding.get_text_embedding("test")
            self.vector_size = len(test_embedding)
            
            logger.info(f"Ollama embedding model loaded successfully")
            logger.info(f"Vector dimensions: {self.vector_size}")
            return True
            
        except ImportError:
            logger.error("llama-index-embeddings-ollama not installed")
            logger.error("  Install with: pip install llama-index-embeddings-ollama")
            return False
        except Exception as e:
            logger.error(f"Failed to load Ollama embedding model: {str(e)}")
            logger.error(f"  Make sure Ollama is running at {config.OLLAMA_URL}")
            logger.error(f"  And model '{self.model_name}' is available")
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if an embedding model can be loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        return self.ollama_embedding is not None
    
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
            embedding = self.ollama_embedding.get_text_embedding(text.strip())
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
        Generate embeddings for multiple texts efficiently.
        
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
            # LlamaIndex OllamaEmbedding doesn't have batch method exposed,
            # so we'll process in batches manually
            embeddings = []
            batch_size = config.OLLAMA_EMBEDDING_BATCH_SIZE
            
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                
                if show_progress:
                    logger.info(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
                
                # Get embeddings for batch
                batch_embeddings = [
                    self.ollama_embedding.get_text_embedding(text.strip())
                    for text in batch if text and text.strip()
                ]
                embeddings.extend(batch_embeddings)
            
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
