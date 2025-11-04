"""
Admin API routes for LangGraph Server.

Provides endpoints for:
- Health checks
- Model management
- Collection management
- Data ingestion
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Dict, Any, List

from .models_api import get_health_status, list_ollama_models, pull_ollama_model, get_models_status
from .collection_api import (
    list_collections,
    get_collection_stats,
    delete_collection as remove_collection,
    clear_collection as empty_collection,
)

# Create admin router
router = APIRouter(prefix="/api/admin", tags=["admin"])


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Check health status of all services.
    
    Returns:
        Health status of Ollama, LM Studio, Qdrant, and LangGraph
    """
    result = await get_health_status()
    
    # Add LangGraph status (always healthy if this endpoint responds)
    if "services" not in result:
        result["services"] = {}
    
    result["services"]["langgraph"] = {
        "status": "healthy",
        "message": "LangGraph server is running",
    }
    
    return result


@router.get("/models")
async def list_models() -> List[Dict[str, Any]]:
    """
    List all available Ollama models.
    
    Returns:
        List of models with name, size, and loaded status
    """
    result = await list_ollama_models()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch models"))
    
    # Transform to match frontend expectations
    models = result.get("models", [])
    return [
        {
            "name": model["name"],
            "size": model.get("size", 0),
            "loaded": True,  # Ollama models are loaded if they're in the list
        }
        for model in models
    ]


@router.post("/models/{model_name}/pull")
async def pull_model(model_name: str) -> Dict[str, Any]:
    """
    Pull an Ollama model.
    
    Args:
        model_name: Name of the model to pull (e.g., "llama3.2:3b")
        
    Returns:
        Pull status
    """
    result = await pull_ollama_model(model_name)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to pull model"))
    
    return result


@router.get("/models/{model_name}")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information
    """
    result = await list_ollama_models()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch models"))
    
    # Find the specific model
    models = result.get("models", [])
    for model in models:
        if model["name"] == model_name:
            return {
                "name": model["name"],
                "size": model.get("size", 0),
                "loaded": True,
            }
    
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


# ============================================================================
# Collection Management Endpoints
# ============================================================================

@router.get("/collections")
async def get_all_collections() -> List[Dict[str, Any]]:
    """
    List all Qdrant collections.
    
    Returns:
        List of collections with metadata
    """
    result = await list_collections()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch collections"))
    
    return result.get("collections", [])


@router.get("/collections/{collection_name}")
async def get_collection(collection_name: str) -> Dict[str, Any]:
    """
    Get information about a specific collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Collection details
    """
    result = await get_collection_stats(collection_name)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Collection not found"))
    
    return result.get("stats", {})


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str) -> Dict[str, Any]:
    """
    Delete a collection.
    
    Args:
        collection_name: Name of the collection to delete
        
    Returns:
        Deletion status
    """
    result = await remove_collection(collection_name)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to delete collection"))
    
    return result


@router.post("/collections/{collection_name}/clear")
async def clear_collection(collection_name: str) -> Dict[str, Any]:
    """
    Clear all vectors from a collection.
    
    Args:
        collection_name: Name of the collection to clear
        
    Returns:
        Clear status
    """
    result = await empty_collection(collection_name)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to clear collection"))
    
    return result


# ============================================================================
# Ingestion Endpoints (placeholder for future implementation)
# ============================================================================

@router.post("/ingestion/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    source_type: str = Form(...),
) -> Dict[str, Any]:
    """
    Upload and ingest a file.
    
    Args:
        file: File to upload
        source_type: Type of source (quran, hadith, etc.)
        
    Returns:
        Ingestion status
    """
    # TODO: Implement file ingestion
    raise HTTPException(status_code=501, detail="File ingestion not yet implemented")


@router.get("/ingestion/status/{task_id}")
async def get_ingestion_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of an ingestion task.
    
    Args:
        task_id: ID of the ingestion task
        
    Returns:
        Task status
    """
    # TODO: Implement task status tracking
    raise HTTPException(status_code=501, detail="Ingestion status tracking not yet implemented")


# Export router
__all__ = ["router"]

