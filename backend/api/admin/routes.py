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

from .models_api import (
    get_health_status,
    list_ollama_models,
    pull_ollama_model,
    get_models_status,
    list_llamacpp_models,
)
from .collection_api import (
    list_collections,
    get_collection_stats,
    delete_collection as remove_collection,
    clear_collection as empty_collection,
)
from .ingestion_api import (
    upload_and_ingest_handler,
    get_ingestion_status,
    list_tasks as list_ingestion_tasks,
    cancel_task as cancel_ingestion_task,
    list_available_files,
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
    List all available models (llama.cpp or Ollama based on configured backend).
    
    Returns:
        List of models with name, type, and status
    """
    from backend.core.config import config
    
    # Return llama.cpp models if that's the configured backend
    if config.LLM_BACKEND == "llamacpp" or config.EMBEDDING_BACKEND == "llamacpp":
        result = await list_llamacpp_models()
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch models"))
        
        return result.get("models", [])
    
    # Fall back to Ollama models
    result = await list_ollama_models()
    
    if not result.get("success"):
        # If Ollama is not available, return empty list instead of error
        # This allows the frontend to gracefully handle when Ollama is not running
        return []
    
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
# Ingestion Endpoints
# ============================================================================

@router.post("/ingestion/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    collection_name: str = Form(None),
    batch_size: int = Form(100),
) -> Dict[str, Any]:
    """
    Upload and ingest a file.
    
    Args:
        file: JSON file to upload and ingest
        source_type: Type of source (quran, hadith, tafsir, fiqh, seerah)
        collection_name: Optional target collection (uses default if not provided)
        batch_size: Batch size for processing (default: 100)
        
    Returns:
        Task information with task_id for status polling
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Process upload
        result = await upload_and_ingest_handler(
            file_content=file_content,
            filename=file.filename,
            source_type=source_type,
            collection_name=collection_name,
            batch_size=batch_size,
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/ingestion/status/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of an ingestion task.
    
    Args:
        task_id: ID of the ingestion task
        
    Returns:
        Task status with progress information
    """
    result = await get_ingestion_status(task_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Task not found"))
    
    return result


@router.get("/ingestion/tasks")
async def list_tasks(limit: int = 50) -> Dict[str, Any]:
    """
    List all ingestion tasks.
    
    Args:
        limit: Maximum number of tasks to return
        
    Returns:
        List of tasks with their status
    """
    result = await list_ingestion_tasks(limit=limit)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to list tasks"))
    
    return result


@router.post("/ingestion/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a queued ingestion task.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Cancellation status
    """
    result = await cancel_ingestion_task(task_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to cancel task"))
    
    return result


@router.get("/ingestion/files")
async def list_files() -> Dict[str, Any]:
    """
    List available data files for ingestion.
    
    Returns:
        Dictionary of available files organized by type
    """
    result = list_available_files()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to list files"))
    
    return result


# Export router
__all__ = ["router"]

