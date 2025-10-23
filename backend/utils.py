"""
Shared utility functions for the Islamic Chatbot RAG system.
"""

import re
import uuid
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import wraps
from pathlib import Path


def setup_logging(name: str = "alimai", level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


def timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"⏱️  {func.__name__} took {elapsed_time:.2f}s")
        return result
    return wrapper


def generate_uuid(namespace: str, name: str) -> str:
    """
    Generate a deterministic UUID based on namespace and name.
    
    Args:
        namespace: Namespace string (e.g., "quran", "hadith")
        name: Unique identifier (e.g., "2:255", "bukhari:1")
        
    Returns:
        UUID string
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}-{name}"))


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_verse_reference(surah_number: int, verse_number: int, surah_name: Optional[str] = None) -> str:
    """
    Format a Quranic verse reference.
    
    Args:
        surah_number: Surah number
        verse_number: Verse number
        surah_name: Optional surah name
        
    Returns:
        Formatted reference (e.g., "Surah Al-Baqarah (2:255)")
    """
    if surah_name:
        return f"Surah {surah_name} ({surah_number}:{verse_number})"
    return f"{surah_number}:{verse_number}"


def format_hadith_reference(collection: str, book: Optional[str] = None, hadith_number: Optional[int] = None) -> str:
    """
    Format a Hadith reference.
    
    Args:
        collection: Hadith collection name (e.g., "Sahih al-Bukhari")
        book: Optional book name
        hadith_number: Optional hadith number
        
    Returns:
        Formatted reference
    """
    parts = [collection]
    if book:
        parts.append(f"Book: {book}")
    if hadith_number:
        parts.append(f"Hadith {hadith_number}")
    return ", ".join(parts)


def extract_arabic_text(text: str) -> str:
    """
    Extract Arabic text from mixed text.
    
    Args:
        text: Input text
        
    Returns:
        Extracted Arabic text
    """
    # Arabic Unicode range: \u0600-\u06FF
    arabic_chars = re.findall(r'[\u0600-\u06FF\s]+', text)
    return ' '.join(arabic_chars).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def safe_dict_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with nested key support.
    
    Args:
        data: Dictionary
        key: Key (supports dot notation for nested keys, e.g., "metadata.surah_number")
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    keys = key.split('.')
    value = data
    
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
            if value is None:
                return default
        else:
            return default
    
    return value if value is not None else default


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO string.
    
    Args:
        dt: Datetime object (defaults to current time)
        
    Returns:
        ISO format string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    import json
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")


def save_json_file(data: Any, file_path: str, indent: int = 2):
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation level
    """
    import json
    
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text.
    
    Uses simple heuristic: ~4 characters per token for English, ~2 for Arabic.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(text) - arabic_chars
    
    # Rough estimates
    arabic_tokens = arabic_chars / 2
    english_tokens = english_chars / 4
    
    return int(arabic_tokens + english_tokens)


class ProgressTracker:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {self.current}/{self.total} ({percentage:.1f}%) - {rate:.1f} items/s", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def finish(self):
        """Mark as complete."""
        self.current = self.total
        self._print_progress()

