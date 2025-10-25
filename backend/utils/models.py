"""
Pydantic models and schemas for the Islamic Chatbot RAG system.

This module defines data models for:
- RAG state management (for LangGraph)
- API requests and responses
- Document and chunk metadata
- Qdrant payloads
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class SourceType(str, Enum):
    """Types of Islamic text sources."""
    QURAN = "quran"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    FIQH = "fiqh"
    SEERAH = "seerah"
    AQIDAH = "aqidah"
    USUL = "usul"


class AuthenticityGrade(str, Enum):
    """Hadith authenticity grades."""
    SAHIH = "sahih"
    HASAN = "hasan"
    DAIF = "daif"
    MAWDU = "mawdu"
    UNKNOWN = "unknown"


class Madhab(str, Enum):
    """Islamic schools of jurisprudence."""
    HANAFI = "hanafi"
    MALIKI = "maliki"
    SHAFI = "shafi"
    HANBALI = "hanbali"


class QuestionType(str, Enum):
    """Types of user questions."""
    FIQH = "fiqh"
    AQIDAH = "aqidah"
    TAFSIR = "tafsir"
    HADITH = "hadith"
    GENERAL = "general"


class ChunkType(str, Enum):
    """Types of text chunks."""
    VERSE = "verse"
    TAFSIR = "tafsir"
    HADITH = "hadith"
    FIQH_RULING = "fiqh_ruling"
    SEERAH_EVENT = "seerah_event"


# ============================================================================
# Document and Chunk Models
# ============================================================================

class SourceMetadata(BaseModel):
    """Source-specific metadata (nested object in Qdrant payload)."""
    
    # Hadith-specific
    hadith_number: Optional[int] = None
    book_name: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_name: Optional[str] = None
    authenticity_grade: Optional[AuthenticityGrade] = None
    narrator_chain: Optional[str] = None
    
    # Quran/Tafsir-specific
    surah_number: Optional[int] = None
    verse_number: Optional[int] = None
    verse_key: Optional[str] = None
    surah_name: Optional[str] = None
    surah_name_arabic: Optional[str] = None
    chunk_type: Optional[ChunkType] = None
    chunk_index: Optional[int] = None
    tafsir_source: Optional[str] = None
    
    # Fiqh-specific
    madhab: Optional[Madhab] = None
    ruling_category: Optional[str] = None
    
    # Seerah-specific
    event_name: Optional[str] = None
    chronological_order: Optional[int] = None
    year_hijri: Optional[int] = None
    
    class Config:
        use_enum_values = True


class References(BaseModel):
    """Cross-references to related content."""
    related_verses: List[str] = Field(default_factory=list)
    related_hadiths: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)


class QdrantPayload(BaseModel):
    """
    Universal payload schema for Qdrant points.
    
    This schema supports all Islamic text types with a compact core + flexible metadata approach.
    """
    
    # Core fields (all sources have these)
    source_type: SourceType
    book_title: str
    book_title_arabic: Optional[str] = None
    author: str
    text_content: str  # Main searchable text
    arabic_text: Optional[str] = None
    english_text: Optional[str] = None
    topic_tags: List[str] = Field(default_factory=list)
    
    # Source-specific metadata (nested, only populated as needed)
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata)
    
    # Cross-references
    references: References = Field(default_factory=References)
    
    # Legacy fields for backward compatibility (optional)
    source: Optional[str] = None
    source_detail: Optional[str] = None
    chapter_name: Optional[str] = None
    chapter_name_arabic: Optional[str] = None
    verse_key: Optional[str] = None
    chapter_number: Optional[int] = None
    verse_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """Represents a single chunk of text with metadata."""
    id: str
    text_content: str
    metadata: QdrantPayload
    vector: Optional[List[float]] = None
    score: Optional[float] = None


# ============================================================================
# RAG State Models (for LangGraph)
# ============================================================================

class Message(BaseModel):
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RAGState(BaseModel):
    """
    State for the RAG workflow in LangGraph.
    
    This represents the state that flows through the LangGraph nodes.
    """
    # User input
    user_query: str
    
    # Query classification
    question_type: Optional[QuestionType] = None
    
    # Expanded queries
    expanded_queries: List[str] = Field(default_factory=list)
    
    # Retrieved documents
    retrieved_docs: List[DocumentChunk] = Field(default_factory=list)
    
    # Ranked and filtered documents
    ranked_docs: List[DocumentChunk] = Field(default_factory=list)
    
    # Generated response
    response: Optional[str] = None
    
    # Formatted citations
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Conversation history
    messages: List[Message] = Field(default_factory=list)
    
    # Metadata
    madhab_preference: Optional[Madhab] = None
    max_sources: int = 10
    score_threshold: float = 0.7
    
    class Config:
        use_enum_values = True


# ============================================================================
# API Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str
    limit: int = 10
    score_threshold: Optional[float] = None
    source_type: Optional[SourceType] = None
    madhab: Optional[Madhab] = None


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str
    results: List[DocumentChunk]
    total_results: int
    processing_time: float


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    message: str
    thread_id: Optional[str] = None
    madhab_preference: Optional[Madhab] = None
    max_sources: int = 10


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    response: str
    sources: List[DocumentChunk]
    thread_id: str
    processing_time: float


class IngestionRequest(BaseModel):
    """Request model for data ingestion."""
    file_path: str
    source_type: SourceType
    batch_size: int = 100
    recreate_collection: bool = False


class IngestionResponse(BaseModel):
    """Response model for data ingestion."""
    source_type: SourceType
    total_documents: int
    total_chunks: int
    uploaded: int
    failed: int
    success_rate: float
    elapsed_time: float


class CollectionStats(BaseModel):
    """Statistics about a Qdrant collection."""
    collection_name: str
    status: str
    points_count: int
    indexed_vectors_count: int
    segments_count: int
    vector_size: Optional[int] = None
    distance_metric: Optional[str] = None


# ============================================================================
# Utility Functions
# ============================================================================

def create_qdrant_payload(
    source_type: SourceType,
    book_title: str,
    author: str,
    text_content: str,
    **kwargs
) -> QdrantPayload:
    """
    Helper function to create a Qdrant payload with proper structure.
    
    Args:
        source_type: Type of source (quran, hadith, etc.)
        book_title: Title of the book/source
        author: Author of the text
        text_content: Main text content
        **kwargs: Additional fields to populate
        
    Returns:
        QdrantPayload instance
    """
    # Extract source metadata fields
    source_metadata_fields = {}
    for key, value in kwargs.items():
        if key in SourceMetadata.model_fields:
            source_metadata_fields[key] = value
    
    # Extract references fields
    references_fields = {}
    for key in ['related_verses', 'related_hadiths', 'related_topics']:
        if key in kwargs:
            references_fields[key] = kwargs[key]
    
    # Build payload
    payload_dict = {
        "source_type": source_type,
        "book_title": book_title,
        "author": author,
        "text_content": text_content,
        "source_metadata": SourceMetadata(**source_metadata_fields),
        "references": References(**references_fields)
    }
    
    # Add optional core fields
    for key in ['book_title_arabic', 'arabic_text', 'english_text', 'topic_tags']:
        if key in kwargs:
            payload_dict[key] = kwargs[key]
    
    return QdrantPayload(**payload_dict)

