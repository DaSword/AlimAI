"""
Admin API endpoints for Qdrant collection management.

Provides endpoints to:
- List collections
- Get collection statistics
- Create/delete collections
- Export collection data
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from backend.vectordb.qdrant_manager import QdrantManager
from backend.core.models import CollectionStats
from backend.core.config import Config


config = Config()


async def list_collections() -> Dict[str, Any]:
    """
    List all Qdrant collections.
    
    Returns:
        Dictionary with collection list
    """
    try:
        manager = QdrantManager(collection_name="temp")
        
        # Get all collections
        collections = manager.client.get_collections()
        
        collection_list = []
        for collection in collections.collections:
            collection_list.append({
                "name": collection.name,
            })
        
        return {
            "success": True,
            "collections": collection_list,
            "total": len(collection_list),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    """
    Get statistics for a specific collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Collection statistics dictionary
    """
    try:
        manager = QdrantManager(collection_name=collection_name)
        
        # Get collection info
        info = manager.client.get_collection(collection_name)
        
        stats = CollectionStats(
            collection_name=collection_name,
            status=info.status.name,
            points_count=info.points_count or 0,
            indexed_vectors_count=info.indexed_vectors_count or 0,
            segments_count=info.segments_count,
            vector_size=info.config.params.vectors.size if info.config.params.vectors else None,
            distance_metric=info.config.params.vectors.distance.name if info.config.params.vectors else None,
        )
        
        return {
            "success": True,
            "stats": stats.model_dump(),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def delete_collection(collection_name: str) -> Dict[str, Any]:
    """
    Delete a collection.
    
    Args:
        collection_name: Name of the collection to delete
        
    Returns:
        Success status
    """
    try:
        manager = QdrantManager(collection_name=collection_name)
        
        # Delete collection
        manager.client.delete_collection(collection_name)
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully",
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def clear_collection(collection_name: str) -> Dict[str, Any]:
    """
    Clear all points from a collection without deleting it.
    
    Args:
        collection_name: Name of the collection to clear
        
    Returns:
        Success status
    """
    try:
        manager = QdrantManager(collection_name=collection_name)
        
        # Get collection info first
        info = manager.client.get_collection(collection_name)
        points_count = info.points_count or 0
        
        # Delete all points
        from qdrant_client.models import FilterSelector
        manager.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=None  # Delete all
            )
        )
        
        return {
            "success": True,
            "message": f"Cleared {points_count} points from collection '{collection_name}'",
            "points_removed": points_count,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def export_collection(
    collection_name: str,
    output_path: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """
    Export collection data to JSON.
    
    Args:
        collection_name: Name of the collection to export
        output_path: Output file path
        limit: Maximum number of points to export
        
    Returns:
        Export result
    """
    try:
        manager = QdrantManager(collection_name=collection_name)
        
        # Scroll through all points
        points, next_offset = manager.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        # Convert to serializable format
        exported_data = []
        for point in points:
            exported_data.append({
                "id": str(point.id),
                "payload": point.payload,
            })
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "message": f"Exported {len(exported_data)} points to {output_path}",
                "points_exported": len(exported_data),
                "file_path": output_path,
            }
        else:
            return {
                "success": True,
                "data": exported_data,
                "points_exported": len(exported_data),
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def search_collection(
    collection_name: str,
    query: str,
    limit: int = 10,
    score_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Search a collection.
    
    Args:
        collection_name: Name of the collection
        query: Search query
        limit: Number of results
        score_threshold: Minimum score threshold
        
    Returns:
        Search results
    """
    try:
        manager = QdrantManager(
            collection_name=collection_name,
            embedding_backend=config.EMBEDDING_BACKEND,
        )
        
        # Search
        results = manager.search(
            query_text=query,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return {
            "success": True,
            "results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ],
            "total_results": len(results),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# Export endpoints
__all__ = [
    "list_collections",
    "get_collection_stats",
    "delete_collection",
    "clear_collection",
    "export_collection",
    "search_collection",
]

