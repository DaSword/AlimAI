"""
Task tracking system for long-running ingestion jobs.

Provides:
- Task state management (pending, running, completed, failed)
- Progress tracking with messages
- Single-job queue (only one ingestion at a time)
- In-memory storage (state lost on server restart)
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict


class TaskStatus(str, Enum):
    """Task execution status."""
    QUEUED = "queued"      # Waiting in queue
    RUNNING = "running"    # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"      # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user


@dataclass
class TaskInfo:
    """Information about a running task."""
    task_id: str
    status: TaskStatus
    progress: float = 0.0  # 0-100
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Task-specific data
    source_type: Optional[str] = None
    file_name: Optional[str] = None
    collection_name: Optional[str] = None
    
    # Results (populated on completion)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        data = asdict(self)
        # Convert datetime to ISO strings
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


class TaskTracker:
    """
    Manages ingestion tasks with a single-job queue.
    
    Only one ingestion can run at a time. Additional requests are queued.
    """
    
    def __init__(self):
        """Initialize the task tracker."""
        self.tasks: Dict[str, TaskInfo] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.current_task_id: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_started = False
    
    def create_task(
        self,
        source_type: str,
        file_name: str,
        collection_name: Optional[str] = None,
    ) -> str:
        """
        Create a new task and add to queue.
        
        Args:
            source_type: Type of source (quran, hadith, etc.)
            file_name: Name of uploaded file
            collection_name: Target collection name
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = TaskInfo(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            source_type=source_type,
            file_name=file_name,
            collection_name=collection_name,
            message="Waiting in queue...",
        )
        
        self.tasks[task_id] = task
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskInfo or None if not found
        """
        return self.tasks.get(task_id)
    
    def list_tasks(self, limit: int = 50) -> list[TaskInfo]:
        """
        List all tasks (most recent first).
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of TaskInfo objects
        """
        tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        return tasks[:limit]
    
    def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str = "",
    ):
        """
        Update task progress.
        
        Args:
            task_id: Task identifier
            progress: Progress percentage (0-100)
            message: Status message
        """
        task = self.tasks.get(task_id)
        if task:
            task.progress = progress
            if message:
                task.message = message
    
    def start_task(self, task_id: str):
        """Mark task as running."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.message = "Starting ingestion..."
            self.current_task_id = task_id
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """
        Mark task as completed successfully.
        
        Args:
            task_id: Task identifier
            result: Ingestion results
        """
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100.0
            task.result = result
            task.message = "Ingestion completed successfully"
            
            if task_id == self.current_task_id:
                self.current_task_id = None
    
    def fail_task(self, task_id: str, error: str):
        """
        Mark task as failed.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            task.message = f"Failed: {error}"
            
            if task_id == self.current_task_id:
                self.current_task_id = None
    
    def cancel_task(self, task_id: str):
        """
        Cancel a queued task.
        
        Args:
            task_id: Task identifier
        """
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.QUEUED:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            task.message = "Cancelled by user"
    
    def is_busy(self) -> bool:
        """
        Check if a task is currently running.
        
        Returns:
            True if a task is running
        """
        return self.current_task_id is not None
    
    def get_queue_position(self, task_id: str) -> Optional[int]:
        """
        Get position in queue (1-indexed).
        
        Args:
            task_id: Task identifier
            
        Returns:
            Position in queue or None if not queued
        """
        queued_tasks = [
            t for t in sorted(self.tasks.values(), key=lambda t: t.created_at)
            if t.status == TaskStatus.QUEUED
        ]
        
        for i, task in enumerate(queued_tasks):
            if task.task_id == task_id:
                return i + 1
        
        return None
    
    async def enqueue_task(
        self,
        task_id: str,
        work_func: Callable,
        *args,
        **kwargs
    ):
        """
        Add task to execution queue.
        
        Args:
            task_id: Task identifier
            work_func: Async function to execute
            *args: Positional arguments for work_func
            **kwargs: Keyword arguments for work_func
        """
        # Ensure worker is running
        if not self._worker_started:
            self._worker_task = asyncio.create_task(self._worker())
            self._worker_started = True
        
        # Add to queue
        await self.queue.put((task_id, work_func, args, kwargs))
    
    async def _worker(self):
        """
        Background worker that processes tasks one at a time.
        
        This ensures only one ingestion runs at a time.
        """
        while True:
            try:
                # Get next task from queue
                task_id, work_func, args, kwargs = await self.queue.get()
                
                task = self.get_task(task_id)
                if not task:
                    continue
                
                # Check if task was cancelled
                if task.status == TaskStatus.CANCELLED:
                    self.queue.task_done()
                    continue
                
                # Mark as running
                self.start_task(task_id)
                
                try:
                    # Execute the work function
                    result = await work_func(*args, **kwargs)
                    
                    # Mark as completed
                    self.complete_task(task_id, result)
                    
                except Exception as e:
                    # Mark as failed
                    self.fail_task(task_id, str(e))
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                # Worker is being shut down
                break
            except Exception as e:
                # Unexpected error in worker
                print(f"Worker error: {e}")
                continue


# Global task tracker instance
task_tracker = TaskTracker()


# Export public API
__all__ = [
    "TaskStatus",
    "TaskInfo",
    "TaskTracker",
    "task_tracker",
]

