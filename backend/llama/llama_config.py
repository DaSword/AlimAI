"""
LlamaIndex Configuration - Global settings for LlamaIndex with embeddings and LLM.

This module configures LlamaIndex with flexible embedding backends:
- HuggingFace/SentenceTransformers (fast, local, batched)
- Ollama API (flexible, model variety)

And Ollama for LLM operations.
"""

from typing import Optional, Union
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from backend.core.config import config
from backend.core.utils import setup_logging

logger = setup_logging("llama_config")


def configure_llama_index(
    embedding_backend: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    lmstudio_url: Optional[str] = None,
    device: Optional[str] = None
) -> None:
    """
    Configure LlamaIndex global settings with flexible embedding and LLM backends.
    
    Args:
        embedding_backend: Embedding backend ('huggingface', 'ollama', or 'lmstudio', defaults to config)
        embedding_model: Embedding model name (defaults to config based on backend)
        llm_backend: LLM backend ('ollama' or 'lmstudio', defaults to config)
        llm_model: LLM model name (defaults to config based on backend)
        ollama_url: Ollama service URL (defaults to config)
        lmstudio_url: LM Studio service URL (defaults to config)
        device: Device for HuggingFace embeddings ('cuda' or 'cpu', auto-detects if None)
    """
    # Use config defaults if not specified
    embedding_backend = embedding_backend or config.EMBEDDING_BACKEND
    llm_backend = llm_backend or config.LLM_BACKEND
    ollama_url = ollama_url or config.OLLAMA_URL
    lmstudio_url = lmstudio_url or config.LMSTUDIO_URL
    
    logger.info("Configuring LlamaIndex...")
    logger.info(f"  Embedding Backend: {embedding_backend}")
    logger.info(f"  LLM Backend: {llm_backend}")
    
    try:
        # Configure embeddings based on backend
        if embedding_backend == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            embedding_model = embedding_model or config.EMBEDDING_MODEL
            
            # Auto-detect device if not specified
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"  Embedding Model: {embedding_model} (device: {device})")

            # Get HuggingFace token from environment if available
            import os
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            
            # Configure HuggingFace embeddings (fast, local, batched)
            embed_model_kwargs = {
                "model_name": embedding_model,
                "device": device,
                "trust_remote_code": True,  # Required for some models
            }
            
            # Add token if available (required for gated models like embeddinggemma)
            if hf_token:
                logger.info("✓ Using HuggingFace token for authentication")
                embed_model_kwargs["token"] = hf_token
            
            embed_model = HuggingFaceEmbedding(**embed_model_kwargs)
            logger.info("✓ Using HuggingFace/SentenceTransformers (fast batching)")
            
        elif embedding_backend == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            embedding_model = embedding_model or config.OLLAMA_EMBEDDING_MODEL
            logger.info(f"  Embedding Model: {embedding_model}")
            
            # Configure Ollama embeddings (flexible, API-based)
            embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url=ollama_url,
                request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
            )
            logger.info("✓ Using Ollama API embeddings")
            
        elif embedding_backend == "lmstudio":
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.embeddings.openai.base import OpenAIEmbeddingMode
            
            embedding_model = embedding_model or config.LMSTUDIO_EMBEDDING_MODEL
            logger.info(f"  Embedding Model: {embedding_model}")
            logger.info(f"  LM Studio URL: {lmstudio_url}")
            
            # Configure LM Studio embeddings (OpenAI-compatible API)
            # Use model_name with SIMILARITY_MODE to support custom models
            embed_model = OpenAIEmbedding(
                model_name=embedding_model,
                api_base=lmstudio_url,
                api_key="lm-studio",  # LM Studio doesn't require a real API key
                mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
            )
            logger.info("✓ Using LM Studio embeddings (OpenAI-compatible)")
            
        else:
            raise ValueError(f"Unknown embedding backend: {embedding_backend}. Use 'huggingface', 'ollama', or 'lmstudio'")
        
        # Configure LLM based on backend
        if llm_backend == "ollama":
            from llama_index.llms.ollama import Ollama
            
            llm_model = llm_model or config.OLLAMA_CHAT_MODEL
            logger.info(f"  LLM Model: {llm_model} (Ollama)")
            
            llm = Ollama(
                model=llm_model,
                base_url=ollama_url,
                request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
            )
            logger.info("✓ Using Ollama LLM")
            
        elif llm_backend == "lmstudio":
            from llama_index.llms.lmstudio import LMStudio
            
            llm_model = llm_model or config.LMSTUDIO_CHAT_MODEL
            logger.info(f"  LLM Model: {llm_model} (LM Studio)")
            logger.info(f"  LM Studio Request Timeout: {config.LMSTUDIO_REQUEST_TIMEOUT}s")
            
            llm = LMStudio(
                model_name=llm_model,
                base_url=lmstudio_url,
                temperature=0.7,
                request_timeout=config.LMSTUDIO_REQUEST_TIMEOUT,  # Use request_timeout, not timeout!
                timeout=config.LMSTUDIO_REQUEST_TIMEOUT,  # Set both for completeness
            )
            logger.info("✓ Using LM Studio LLM")
            
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}. Use 'ollama' or 'lmstudio'")
        
        # Set global defaults
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = config.CHUNK_SIZE_MAX
        Settings.chunk_overlap = 200
        
        logger.info("✓ LlamaIndex configured successfully")
        logger.info(f"✓ Embedding vector size: {len(embed_model.get_text_embedding('test'))}")
        
    except Exception as e:
        logger.error(f"Failed to configure LlamaIndex: {str(e)}")
        raise


def get_embed_model(
    embedding_backend: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None
):
    """
    Get a configured embedding model instance based on backend.
    
    Args:
        embedding_backend: Embedding backend ('huggingface', 'ollama', or 'lmstudio', defaults to config)
        model_name: Embedding model name (defaults to config based on backend)
        device: Device for HuggingFace embeddings ('cuda' or 'cpu', auto-detects if None)
        
    Returns:
        Configured embedding model instance
    """
    embedding_backend = embedding_backend or config.EMBEDDING_BACKEND
    
    if embedding_backend == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        model_name = model_name or config.EMBEDDING_MODEL
        
        # Auto-detect device if not specified
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get HuggingFace token from environment if available
        import os
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        
        embed_model_kwargs = {
            "model_name": model_name,
            "device": device,
            "trust_remote_code": True,
        }
        
        if hf_token:
            embed_model_kwargs["token"] = hf_token
        
        return HuggingFaceEmbedding(**embed_model_kwargs)
    
    elif embedding_backend == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        model_name = model_name or config.OLLAMA_EMBEDDING_MODEL
        
        return OllamaEmbedding(
            model_name=model_name,
            base_url=config.OLLAMA_URL,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
    
    elif embedding_backend == "lmstudio":
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.embeddings.openai.base import OpenAIEmbeddingMode
        
        model_name = model_name or config.LMSTUDIO_EMBEDDING_MODEL
        
        return OpenAIEmbedding(
            model_name=model_name,
            api_base=config.LMSTUDIO_URL,
            api_key="lm-studio",
            mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
        )
    
    else:
        raise ValueError(f"Unknown embedding backend: {embedding_backend}. Use 'huggingface', 'ollama', or 'lmstudio'")


def get_llm(
    llm_backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
):
    """
    Get a configured LLM instance based on backend.
    
    Args:
        llm_backend: LLM backend ('ollama' or 'lmstudio', defaults to config)
        model: LLM model name (defaults to config based on backend)
        base_url: Service URL (defaults to config based on backend)
        temperature: Sampling temperature (0.0 to 1.0)
        **kwargs: Additional arguments for the LLM
        
    Returns:
        Configured LLM instance (Ollama or LMStudio)
    """
    llm_backend = llm_backend or config.LLM_BACKEND
    
    if llm_backend == "ollama":
        from llama_index.llms.ollama import Ollama
        
        model = model or config.OLLAMA_CHAT_MODEL
        base_url = base_url or config.OLLAMA_URL
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
            **kwargs
        )
    
    elif llm_backend == "lmstudio":
        from llama_index.llms.lmstudio import LMStudio
        
        model = model or config.LMSTUDIO_CHAT_MODEL
        base_url = base_url or config.LMSTUDIO_URL
        
        return LMStudio(
            model_name=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=config.LMSTUDIO_REQUEST_TIMEOUT,  # Use request_timeout, not timeout!
            timeout=config.LMSTUDIO_REQUEST_TIMEOUT,  # Set both for completeness
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown LLM backend: {llm_backend}. Use 'ollama' or 'lmstudio'")


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
