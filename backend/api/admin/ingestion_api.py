"""
Admin API endpoints for data ingestion.

Provides endpoints to:
- Upload and ingest files
- Check ingestion status
- Monitor progress
- List available files
"""

import os
import asyncio
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from backend.ingestion.ingestion import IngestionManager
from backend.core.models import SourceType
from backend.core.config import Config
from backend.core.utils import setup_logging
from .task_tracker import task_tracker, TaskStatus

logger = setup_logging("ingestion_api")
config = Config()

# Ensure uploads directory exists
UPLOADS_DIR = Config.DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file to data/uploads directory.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        
    Returns:
        Path to saved file
    """
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    file_path = UPLOADS_DIR / safe_filename
    
    # Save file asynchronously
    def write_file():
        with open(file_path, 'wb') as f:
            f.write(file_content)
    
    await asyncio.to_thread(write_file)
    
    logger.info(f"Saved uploaded file: {file_path}")
    return str(file_path)


def run_ingestion_sync(
    task_id: str,
    file_path: str,
    source_type: SourceType,
    collection_name: Optional[str],
    batch_size: int,
) -> Dict[str, Any]:
    """
    Run ingestion synchronously (called from thread pool).
    
    This is the blocking function that does the actual work.
    
    Args:
        task_id: Task identifier
        file_path: Path to JSON file
        source_type: Type of source
        collection_name: Target collection
        batch_size: Batch size for processing
        
    Returns:
        Ingestion results
    """
    def progress_callback(progress: float, message: str):
        """Update task progress."""
        task_tracker.update_progress(task_id, progress, message)
    
    # Initialize ingestion manager
    manager = IngestionManager()
    
    # Run ingestion with progress tracking
    result = manager.ingest_file(
        file_path=file_path,
        source_type=source_type,
        collection_name=collection_name,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    
    return result


async def run_ingestion_async(
    task_id: str,
    file_path: str,
    source_type: SourceType,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Run ingestion asynchronously using thread pool.
    
    This wraps the blocking ingestion in asyncio.to_thread so it doesn't
    block the ASGI event loop.
    
    Args:
        task_id: Task identifier
        file_path: Path to JSON file
        source_type: Type of source
        collection_name: Target collection
        batch_size: Batch size
        
    Returns:
        Ingestion results
    """
    result = await asyncio.to_thread(
        run_ingestion_sync,
        task_id=task_id,
        file_path=file_path,
        source_type=source_type,
        collection_name=collection_name,
        batch_size=batch_size,
    )
    
    return result


async def upload_and_ingest_handler(
    file_content: bytes,
    filename: str,
    source_type: str,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Handle file upload and ingestion request.
    
    This endpoint:
    1. Saves the uploaded file to data/uploads/
    2. Creates a task in the queue
    3. Returns task_id immediately
    4. Ingestion runs in background
    
    Args:
        file_content: Uploaded file content
        filename: Original filename
        source_type: Type of source (quran, hadith, etc.)
        collection_name: Qdrant collection name
        batch_size: Batch size for ingestion
        
    Returns:
        Task information with task_id
    """
    try:
        # Validate source type
        try:
            source_type_enum = SourceType(source_type)
        except ValueError:
            valid_types = [st.value for st in SourceType]
            return {
                "success": False,
                "error": f"Invalid source type: {source_type}. Valid types: {valid_types}",
            }
        
        # Validate file is JSON
        if not filename.lower().endswith('.json'):
            return {
                "success": False,
                "error": "Only JSON files are supported",
            }
        
        # Save uploaded file
        file_path = await save_uploaded_file(file_content, filename)
        
        # Determine collection name
        if not collection_name:
            collection_name = config.QDRANT_COLLECTION
        
        # Create task
        task_id = task_tracker.create_task(
            source_type=source_type,
            file_name=filename,
            collection_name=collection_name,
        )
        
        # Enqueue for processing
        # Pass work function arguments (task_id goes as positional to avoid name conflict)
        await task_tracker.enqueue_task(
            task_id,
            run_ingestion_async,
            task_id,  # First arg to run_ingestion_async
            file_path,  # Second arg
            source_type_enum,  # Third arg
            collection_name,  # Fourth arg (optional)
            batch_size,  # Fifth arg (optional)
        )
        
        # Get queue position
        position = task_tracker.get_queue_position(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "queued",
            "message": f"File uploaded successfully. Position in queue: {position}",
            "file_name": filename,
            "collection_name": collection_name,
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
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
        Task status dictionary
    """
    task = task_tracker.get_task(task_id)
    
    if not task:
        return {
            "success": False,
            "error": f"Task not found: {task_id}",
        }
    
    response = {
        "success": True,
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "file_name": task.file_name,
        "collection_name": task.collection_name,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
    
    # Add queue position if queued
    if task.status == TaskStatus.QUEUED:
        position = task_tracker.get_queue_position(task_id)
        response["queue_position"] = position
    
    # Add result if completed
    if task.status == TaskStatus.COMPLETED and task.result:
        response["result"] = task.result
    
    # Add error if failed
    if task.status == TaskStatus.FAILED and task.error:
        response["error"] = task.error
    
    return response


async def list_tasks(limit: int = 50) -> Dict[str, Any]:
    """
    List all ingestion tasks.
    
    Args:
        limit: Maximum number of tasks to return
        
    Returns:
        List of tasks
    """
    tasks = task_tracker.list_tasks(limit=limit)
    
    return {
        "success": True,
        "tasks": [task.to_dict() for task in tasks],
        "total": len(tasks),
    }


async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a queued task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Cancellation status
    """
    task = task_tracker.get_task(task_id)
    
    if not task:
        return {
            "success": False,
            "error": f"Task not found: {task_id}",
        }
    
    if task.status != TaskStatus.QUEUED:
        return {
            "success": False,
            "error": f"Cannot cancel task with status: {task.status.value}",
        }
    
    task_tracker.cancel_task(task_id)
    
    return {
        "success": True,
        "message": f"Task {task_id} cancelled",
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
            "uploads": [],
        }
        
        # Scan data directory
        for source_type in files_by_type.keys():
            if source_type == "uploads":
                type_dir = UPLOADS_DIR
            else:
                type_dir = data_path / source_type
                
            if type_dir.exists() and type_dir.is_dir():
                json_files = list(type_dir.glob("*.json"))
                files_by_type[source_type] = [
                    {
                        "name": f.name,
                        "path": str(f),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                    for f in json_files
                ]
        
        # Add root-level files
        root_json = list(data_path.glob("*.json"))
        for f in root_json:
            if "quran" in f.name.lower():
                files_by_type["quran"].append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        
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
    "upload_and_ingest_handler",
    "get_ingestion_status",
    "list_tasks",
    "cancel_task",
    "list_available_files",
]
