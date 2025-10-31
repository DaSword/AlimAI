"""
Admin API endpoints for data ingestion.

Provides endpoints to:
- Trigger data ingestion
- Check ingestion status
- Monitor progress
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path

from backend.ingestion.ingestion import ingest_quran, IngestionManager
from backend.core.models import SourceType, IngestionResponse
from backend.core.config import Config


config = Config()


async def ingest_file_handler(
    file_path: str,
    source_type: str,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
    recreate_collection: bool = False,
) -> Dict[str, Any]:
    """
    Handle file ingestion request.
    
    Args:
        file_path: Path to the JSON file
        source_type: Type of source (quran, hadith, etc.)
        collection_name: Qdrant collection name
        batch_size: Batch size for ingestion
        recreate_collection: Whether to recreate the collection
        
    Returns:
        Ingestion result dictionary
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }
        
        # Validate source type
        try:
            source_type_enum = SourceType(source_type)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid source type: {source_type}",
            }
        
        # Determine collection name
        if not collection_name:
            collection_name = config.QDRANT_COLLECTION
        
        # Initialize ingestion manager
        manager = IngestionManager(
            collection_name=collection_name,
            embedding_backend=config.EMBEDDING_BACKEND,
        )
        
        # Ingest based on source type
        if source_type_enum == SourceType.QURAN:
            result = await ingest_quran(
                file_path=file_path,
                collection_name=collection_name,
                batch_size=batch_size,
                recreate=recreate_collection,
            )
        else:
            # Generic ingestion for other types
            result = await manager.ingest_from_json(
                file_path=file_path,
                source_type=source_type_enum,
                batch_size=batch_size,
                recreate_collection=recreate_collection,
            )
        
        return {
            "success": True,
            "result": result,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_ingestion_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of an ingestion task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Status dictionary
    """
    # This would typically query a task queue/database
    # For now, return a placeholder
    return {
        "task_id": task_id,
        "status": "completed",  # pending, running, completed, failed
        "progress": 100,
        "message": "Ingestion task tracking not yet implemented",
    }


def list_available_files(data_dir: str = "data") -> Dict[str, Any]:
    """
    List available data files for ingestion.
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary of available files by type
    """
    try:
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {
                "success": False,
                "error": f"Data directory not found: {data_dir}",
            }
        
        files_by_type = {
            "quran": [],
            "hadith": [],
            "tafsir": [],
            "fiqh": [],
            "seerah": [],
            "aqidah": [],
        }
        
        # Scan data directory
        for source_type in files_by_type.keys():
            type_dir = data_path / source_type
            if type_dir.exists() and type_dir.is_dir():
                json_files = list(type_dir.glob("*.json"))
                files_by_type[source_type] = [str(f) for f in json_files]
        
        # Add root-level files
        root_json = list(data_path.glob("*.json"))
        files_by_type["quran"].extend([str(f) for f in root_json if "quran" in f.name.lower()])
        
        return {
            "success": True,
            "files": files_by_type,
            "total_files": sum(len(files) for files in files_by_type.values()),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# Export endpoints
__all__ = [
    "ingest_file_handler",
    "get_ingestion_status",
    "list_available_files",
]

