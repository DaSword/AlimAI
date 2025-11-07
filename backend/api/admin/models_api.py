"""
Admin API endpoints for model management.

Provides endpoints to:
- Check model status
- Health checks for services
- Get model information
"""

from typing import Dict, Any, List
import httpx

from backend.core.config import Config
from backend.llama.llama_config import check_ollama_connection, check_model_available


config = Config()


async def get_models_status() -> Dict[str, Any]:
    """
    Get status of all configured models.
    
    Returns:
        Model status dictionary
    """
    try:
        models_status = {
            "embedding_backend": config.EMBEDDING_BACKEND,
            "llm_backend": config.LLM_BACKEND,
            "models": {},
        }
        
        # Check embedding model
        if config.EMBEDDING_BACKEND == "huggingface":
            models_status["models"]["embedding"] = {
                "backend": "huggingface",
                "model": config.HF_EMBEDDING_MODEL,
                "status": "available",  # HuggingFace models are always available once downloaded
            }
        elif config.EMBEDDING_BACKEND == "ollama":
            embedding_available = check_model_available(config.OLLAMA_EMBEDDING_MODEL)
            models_status["models"]["embedding"] = {
                "backend": "ollama",
                "model": config.OLLAMA_EMBEDDING_MODEL,
                "status": "available" if embedding_available else "not_loaded",
            }
        elif config.EMBEDDING_BACKEND == "lmstudio":
            models_status["models"]["embedding"] = {
                "backend": "lmstudio",
                "model": config.LMSTUDIO_EMBEDDING_MODEL,
                "status": "unknown",  # Would need to check LM Studio API
            }
        
        # Check LLM model
        if config.LLM_BACKEND == "ollama":
            llm_available = check_model_available(config.OLLAMA_LLM_MODEL)
            models_status["models"]["llm"] = {
                "backend": "ollama",
                "model": config.OLLAMA_LLM_MODEL,
                "status": "available" if llm_available else "not_loaded",
            }
        elif config.LLM_BACKEND == "lmstudio":
            models_status["models"]["llm"] = {
                "backend": "lmstudio",
                "model": config.LMSTUDIO_LLM_MODEL,
                "status": "unknown",  # Would need to check LM Studio API
            }
        
        return {
            "success": True,
            **models_status,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_health_status() -> Dict[str, Any]:
    """
    Get health status of all services.
    
    Returns:
        Health status dictionary
    """
    health = {
        "services": {},
        "overall_status": "healthy",
    }
    
    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.QDRANT_URL}/healthz", timeout=5.0)
            health["services"]["qdrant"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": config.QDRANT_URL,
            }
    except Exception as e:
        health["services"]["qdrant"] = {
            "status": "unreachable",
            "error": str(e),
        }
        health["overall_status"] = "degraded"
    
    # Check Ollama (if configured)
    if config.EMBEDDING_BACKEND == "ollama" or config.LLM_BACKEND == "ollama":
        ollama_healthy = check_ollama_connection()
        health["services"]["ollama"] = {
            "status": "healthy" if ollama_healthy else "unreachable",
            "url": config.OLLAMA_URL,
        }
        if not ollama_healthy:
            health["overall_status"] = "degraded"
    
    # Check LM Studio (if configured)
    if config.EMBEDDING_BACKEND == "lmstudio" or config.LLM_BACKEND == "lmstudio":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{config.LMSTUDIO_URL}/models", timeout=5.0)
                health["services"]["lmstudio"] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": config.LMSTUDIO_URL,
                }
        except Exception as e:
            health["services"]["lmstudio"] = {
                "status": "unreachable",
                "error": str(e),
            }
            health["overall_status"] = "degraded"
    
    return {
        "success": True,
        **health,
    }


async def list_ollama_models() -> Dict[str, Any]:
    """
    List all available Ollama models.
    
    Returns:
        List of Ollama models
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.OLLAMA_URL}/api/tags",
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                return {
                    "success": True,
                    "models": [
                        {
                            "name": model["name"],
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at"),
                        }
                        for model in models
                    ],
                    "total": len(models),
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API returned status {response.status_code}",
                }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def pull_ollama_model(model_name: str) -> Dict[str, Any]:
    """
    Pull an Ollama model.
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        Pull status
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.OLLAMA_URL}/api/pull",
                json={"name": model_name},
                timeout=300.0,  # 5 minutes for large models
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Successfully pulled model '{model_name}'",
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to pull model: {response.text}",
                }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# Export endpoints
__all__ = [
    "get_models_status",
    "get_health_status",
    "list_ollama_models",
    "pull_ollama_model",
]

