"""
LlamaIndex Configuration - Global settings for LlamaIndex with Ollama integration.

This module configures LlamaIndex to use Ollama for embeddings and LLM operations.
It sets up global Settings that will be used throughout the application.
"""

from typing import Optional
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from backend.config import config
from backend.utils import setup_logging

logger = setup_logging("llama_config")


def configure_llama_index(
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    ollama_url: Optional[str] = None
) -> None:
    """
    Configure LlamaIndex global settings with Ollama embeddings and LLM.
    
    Args:
        embedding_model: Ollama embedding model name (defaults to config)
        llm_model: Ollama chat model name (defaults to config)
        ollama_url: Ollama service URL (defaults to config)
    """
    # Use config defaults if not specified
    embedding_model = embedding_model or config.OLLAMA_EMBEDDING_MODEL
    llm_model = llm_model or config.OLLAMA_CHAT_MODEL
    ollama_url = ollama_url or config.OLLAMA_URL
    
    logger.info("Configuring LlamaIndex with Ollama...")
    logger.info(f"  Embedding Model: {embedding_model}")
    logger.info(f"  LLM Model: {llm_model}")
    logger.info(f"  Ollama URL: {ollama_url}")
    
    try:
        # Configure Ollama embeddings
        embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_url,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        
        # Configure Ollama LLM
        llm = Ollama(
            model=llm_model,
            base_url=ollama_url,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        
        # Set global defaults
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = config.CHUNK_SIZE_MAX
        Settings.chunk_overlap = 200
        
        logger.info("LlamaIndex configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to configure LlamaIndex: {str(e)}")
        raise


def get_embed_model(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> OllamaEmbedding:
    """
    Get a configured OllamaEmbedding instance.
    
    Args:
        model_name: Ollama embedding model name
        base_url: Ollama service URL
        
    Returns:
        Configured OllamaEmbedding instance
    """
    model_name = model_name or config.OLLAMA_EMBEDDING_MODEL
    base_url = base_url or config.OLLAMA_URL
    
    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )


def get_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Ollama:
    """
    Get a configured Ollama LLM instance.
    
    Args:
        model: Ollama chat model name
        base_url: Ollama service URL
        temperature: Sampling temperature (0.0 to 1.0)
        **kwargs: Additional arguments for Ollama
        
    Returns:
        Configured Ollama instance
    """
    model = model or config.OLLAMA_CHAT_MODEL
    base_url = base_url or config.OLLAMA_URL
    
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        **kwargs
    )


def check_ollama_connection() -> bool:
    """
    Check if Ollama service is reachable.
    
    Returns:
        True if Ollama is reachable, False otherwise
    """
    import httpx
    
    try:
        response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            logger.info("Ollama service is reachable")
            models = response.json().get("models", [])
            logger.info(f"Available models: {[m['name'] for m in models]}")
            return True
        else:
            logger.warning(f"Ollama returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Cannot reach Ollama service: {str(e)}")
        logger.error(f"   Make sure Ollama is running at {config.OLLAMA_URL}")
        return False


def list_ollama_models() -> list:
    """
    List all available Ollama models.
    
    Returns:
        List of model information dictionaries
    """
    import httpx
    
    try:
        response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        logger.error(f"Error listing Ollama models: {str(e)}")
        return []


def check_model_available(model_name: str) -> bool:
    """
    Check if a specific model is available in Ollama.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    models = list_ollama_models()
    available_names = [m["name"] for m in models]
    
    # Check exact match or with :latest tag
    is_available = (
        model_name in available_names or
        f"{model_name}:latest" in available_names or
        any(m["name"].startswith(f"{model_name}:") for m in models)
    )
    
    if is_available:
        logger.info(f"Model '{model_name}' is available")
    else:
        logger.warning(f"Model '{model_name}' not found in Ollama")
        logger.info(f"   Available models: {available_names}")
        
    return is_available


if __name__ == "__main__":
    """Test the LlamaIndex configuration."""
    
    # Check Ollama connection
    print("\n" + "="*60)
    print("Testing Ollama Connection")
    print("="*60)
    
    if not check_ollama_connection():
        print("Cannot connect to Ollama. Make sure it's running.")
        exit(1)
    
    # Check required models
    print("\n" + "="*60)
    print("Checking Required Models")
    print("="*60)
    
    required_models = [
        config.OLLAMA_EMBEDDING_MODEL,
        config.OLLAMA_CHAT_MODEL,
    ]
    
    all_available = True
    for model in required_models:
        if not check_model_available(model):
            all_available = False
    
    if not all_available:
        print("\nSome models are missing. Pull them with:")
        print(f"  ollama pull {config.OLLAMA_EMBEDDING_MODEL}")
        print(f"  ollama pull {config.OLLAMA_CHAT_MODEL}")
    else:
        print("\nAll required models are available!")
    
    # Configure LlamaIndex
    print("\n" + "="*60)
    print("Configuring LlamaIndex")
    print("="*60)
    
    try:
        configure_llama_index()
        print("LlamaIndex configuration successful!")
    except Exception as e:
        print(f"LlamaIndex configuration failed: {e}")
        exit(1)
