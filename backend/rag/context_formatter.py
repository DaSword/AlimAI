"""
Context formatting and ranking for RAG pipeline.

This module handles:
- Ranking sources by authenticity (Quran > Sahih Hadith > Fiqh)
- Formatting context for LLM prompts
- Grouping sources by type
- Creating structured context templates
"""

from typing import List, Dict, Any, Optional
from backend.core.models import (
    DocumentChunk, 
    SourceType, 
    QuestionType, 
    AuthenticityGrade
)


# ============================================================================
# Authenticity Weighting
# ============================================================================

AUTHENTICITY_WEIGHTS = {
    # Source type base weights
    SourceType.QURAN: 1.0,      # Highest authority
    SourceType.HADITH: 0.85,    # High authority (varies by grade)
    SourceType.TAFSIR: 0.70,    # Scholarly interpretation
    SourceType.FIQH: 0.65,      # Jurisprudential rulings
    SourceType.AQIDAH: 0.75,    # Theological texts
    SourceType.SEERAH: 0.60,    # Historical biography
    SourceType.USUL: 0.70,      # Methodology
}

# Hadith authenticity multipliers
HADITH_GRADE_MULTIPLIERS = {
    AuthenticityGrade.SAHIH: 1.0,
    AuthenticityGrade.HASAN: 0.85,
    AuthenticityGrade.DAIF: 0.50,
    AuthenticityGrade.MAWDU: 0.20,
    AuthenticityGrade.UNKNOWN: 0.70,
}


def calculate_authenticity_score(doc: DocumentChunk) -> float:
    """
    Calculate authenticity score for a document.
    
    Args:
        doc: Document chunk with metadata
        
    Returns:
        Authenticity score (0.0 to 1.0)
    """
    base_weight = AUTHENTICITY_WEIGHTS.get(doc.metadata.source_type, 0.5)
    
    # Apply hadith grade multiplier if applicable
    if doc.metadata.source_type == SourceType.HADITH:
        grade = doc.metadata.source_metadata.authenticity_grade
        if grade:
            multiplier = HADITH_GRADE_MULTIPLIERS.get(grade, 0.7)
            return base_weight * multiplier
    
    return base_weight


def calculate_final_score(doc: DocumentChunk) -> float:
    """
    Calculate final ranking score combining similarity and authenticity.
    
    Args:
        doc: Document chunk with similarity score
        
    Returns:
        Final combined score
    """
    similarity_score = doc.score if doc.score is not None else 0.5
    authenticity_score = calculate_authenticity_score(doc)
    
    # Weighted combination: 60% similarity, 40% authenticity
    final_score = (similarity_score * 0.6) + (authenticity_score * 0.4)
    
    return final_score


# ============================================================================
# Document Ranking
# ============================================================================

def rank_documents(
    documents: List[DocumentChunk],
    question_type: Optional[QuestionType] = None,
    prioritize_authenticity: bool = True
) -> List[DocumentChunk]:
    """
    Rank documents by relevance and authenticity.
    
    Args:
        documents: List of retrieved documents
        question_type: Type of question (for specialized ranking)
        prioritize_authenticity: Whether to weight authenticity heavily
        
    Returns:
        Ranked list of documents
    """
    if not documents:
        return []
    
    # Calculate final scores
    scored_docs = []
    for doc in documents:
        final_score = calculate_final_score(doc)
        
        # Boost Quran and Sahih Hadith for aqidah questions
        if question_type == QuestionType.AQIDAH:
            if doc.metadata.source_type == SourceType.QURAN:
                final_score *= 1.2
            elif (doc.metadata.source_type == SourceType.HADITH and 
                  doc.metadata.source_metadata.authenticity_grade == AuthenticityGrade.SAHIH):
                final_score *= 1.15
        
        scored_docs.append((doc, final_score))
    
    # Sort by final score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs]


def deduplicate_documents(
    documents: List[DocumentChunk],
    similarity_threshold: float = 0.95
) -> List[DocumentChunk]:
    """
    Remove duplicate or near-duplicate documents.
    
    Args:
        documents: List of documents
        similarity_threshold: Text similarity threshold for deduplication
        
    Returns:
        Deduplicated list of documents
    """
    if not documents:
        return []
    
    unique_docs = []
    seen_texts = set()
    
    for doc in documents:
        # Simple deduplication based on text content
        text_key = doc.text_content.strip().lower()[:200]  # First 200 chars
        
        if text_key not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(text_key)
    
    return unique_docs


# ============================================================================
# Source Grouping
# ============================================================================

def group_documents_by_source_type(
    documents: List[DocumentChunk]
) -> Dict[SourceType, List[DocumentChunk]]:
    """
    Group documents by source type.
    
    Args:
        documents: List of documents
        
    Returns:
        Dict mapping source type to list of documents
    """
    grouped = {}
    
    for doc in documents:
        source_type = doc.metadata.source_type
        if source_type not in grouped:
            grouped[source_type] = []
        grouped[source_type].append(doc)
    
    return grouped


def group_fiqh_by_madhab(
    documents: List[DocumentChunk]
) -> Dict[str, List[DocumentChunk]]:
    """
    Group fiqh documents by madhab (school of jurisprudence).
    
    Args:
        documents: List of fiqh documents
        
    Returns:
        Dict mapping madhab name to list of documents
    """
    grouped = {}
    
    for doc in documents:
        if doc.metadata.source_type == SourceType.FIQH:
            madhab = doc.metadata.source_metadata.madhab
            if madhab:
                madhab_name = str(madhab).title()
                if madhab_name not in grouped:
                    grouped[madhab_name] = []
                grouped[madhab_name].append(doc)
    
    return grouped


# ============================================================================
# Context Formatting
# ============================================================================

def format_document_for_context(doc: DocumentChunk, include_metadata: bool = True) -> str:
    """
    Format a single document for inclusion in context.
    
    Args:
        doc: Document chunk
        include_metadata: Whether to include source metadata
        
    Returns:
        Formatted document string
    """
    lines = []
    
    # Source header
    source_type = str(doc.metadata.source_type).upper()
    book_title = doc.metadata.book_title
    
    if doc.metadata.source_type == SourceType.QURAN:
        verse_key = doc.metadata.source_metadata.verse_key or doc.metadata.verse_key
        header = f"[QURAN] {book_title} - {verse_key}"
    elif doc.metadata.source_type == SourceType.HADITH:
        hadith_num = doc.metadata.source_metadata.hadith_number
        authenticity = doc.metadata.source_metadata.authenticity_grade
        header = f"[HADITH] {book_title}, Hadith {hadith_num} ({authenticity})"
    elif doc.metadata.source_type == SourceType.TAFSIR:
        tafsir_source = doc.metadata.source_metadata.tafsir_source or "Commentary"
        verse_key = doc.metadata.source_metadata.verse_key
        header = f"[TAFSIR] {tafsir_source} on {verse_key}"
    elif doc.metadata.source_type == SourceType.FIQH:
        madhab = doc.metadata.source_metadata.madhab
        header = f"[FIQH - {madhab}] {book_title}"
    else:
        header = f"[{source_type}] {book_title}"
    
    lines.append(header)
    
    # Content
    if doc.metadata.arabic_text:
        lines.append(f"Arabic: {doc.metadata.arabic_text}")
    if doc.metadata.english_text:
        lines.append(f"Translation: {doc.metadata.english_text}")
    elif doc.text_content:
        lines.append(f"Text: {doc.text_content}")
    
    # Metadata
    if include_metadata:
        metadata_parts = []
        
        # Add book/chapter info
        if doc.metadata.source_metadata.book_name:
            metadata_parts.append(f"Book: {doc.metadata.source_metadata.book_name}")
        if doc.metadata.source_metadata.chapter_name:
            metadata_parts.append(f"Chapter: {doc.metadata.source_metadata.chapter_name}")
        
        # Add authenticity score
        auth_score = calculate_authenticity_score(doc)
        metadata_parts.append(f"Authenticity: {auth_score:.2f}")
        
        if metadata_parts:
            lines.append(f"Metadata: {', '.join(metadata_parts)}")
    
    return "\n".join(lines)


def format_grouped_context(
    grouped_docs: Dict[SourceType, List[DocumentChunk]],
    max_per_type: int = 5
) -> str:
    """
    Format grouped documents into structured context.
    
    Args:
        grouped_docs: Documents grouped by source type
        max_per_type: Maximum documents per source type
        
    Returns:
        Formatted context string
    """
    sections = []
    
    # Define section order (by authenticity)
    section_order = [
        SourceType.QURAN,
        SourceType.HADITH,
        SourceType.TAFSIR,
        SourceType.AQIDAH,
        SourceType.FIQH,
        SourceType.SEERAH,
        SourceType.USUL,
    ]
    
    for source_type in section_order:
        if source_type not in grouped_docs:
            continue
        
        docs = grouped_docs[source_type][:max_per_type]
        if not docs:
            continue
        
        # Section header
        section_name = str(source_type).upper().replace('_', ' ')
        sections.append(f"\n{'='*60}")
        sections.append(f"{section_name} SOURCES")
        sections.append(f"{'='*60}\n")
        
        # Add documents
        for i, doc in enumerate(docs, 1):
            sections.append(f"\n--- Source {i} ---")
            sections.append(format_document_for_context(doc, include_metadata=True))
        
    return "\n".join(sections)


def format_context_for_llm(
    documents: List[DocumentChunk],
    question_type: Optional[QuestionType] = None,
    max_sources: int = 10
) -> str:
    """
    Format documents into context optimized for LLM consumption.
    
    Args:
        documents: List of retrieved documents
        question_type: Type of question (for specialized formatting)
        max_sources: Maximum number of sources to include
        
    Returns:
        Formatted context string for LLM
    """
    if not documents:
        return "No relevant sources found."
    
    # Rank and deduplicate
    ranked_docs = rank_documents(documents, question_type)
    unique_docs = deduplicate_documents(ranked_docs)
    
    # Limit to max sources
    selected_docs = unique_docs[:max_sources]
    
    # Group by source type
    grouped = group_documents_by_source_type(selected_docs)
    
    # Format based on question type
    if question_type == QuestionType.FIQH and SourceType.FIQH in grouped:
        # Special formatting for fiqh - show madhab perspectives
        return format_fiqh_context(selected_docs)
    else:
        # Standard formatting
        return format_grouped_context(grouped)


def format_fiqh_context(documents: List[DocumentChunk]) -> str:
    """
    Format fiqh documents with madhab perspectives.
    
    Args:
        documents: List of fiqh and related documents
        
    Returns:
        Formatted fiqh context with madhab breakdown
    """
    sections = []
    
    # Separate fiqh from other sources
    fiqh_docs = [d for d in documents if d.metadata.source_type == SourceType.FIQH]
    other_docs = [d for d in documents if d.metadata.source_type != SourceType.FIQH]
    
    # Primary evidence (Quran/Hadith)
    if other_docs:
        sections.append("PRIMARY EVIDENCE:")
        sections.append("="*60)
        for doc in other_docs[:5]:
            sections.append(format_document_for_context(doc))
            sections.append("")
    
    # Madhab perspectives
    if fiqh_docs:
        madhab_groups = group_fiqh_by_madhab(fiqh_docs)
        
        sections.append("\nJURISPRUDENTIAL VIEWS BY MADHAB:")
        sections.append("="*60)
        
        for madhab_name in ["Hanafi", "Maliki", "Shafi", "Hanbali"]:
            if madhab_name in madhab_groups:
                sections.append(f"\n{madhab_name.upper()} SCHOOL:")
                sections.append("-"*40)
                for doc in madhab_groups[madhab_name][:2]:
                    sections.append(format_document_for_context(doc))
                    sections.append("")
    
    return "\n".join(sections)


def create_citation_list(documents: List[DocumentChunk]) -> List[Dict[str, Any]]:
    """
    Create a structured list of citations from documents.
    
    Args:
        documents: List of documents used in response
        
    Returns:
        List of citation dictionaries
    """
    citations = []
    
    for doc in documents:
        citation = {
            "source_type": str(doc.metadata.source_type),
            "book_title": doc.metadata.book_title,
            "author": doc.metadata.author,
        }
        
        # Add source-specific citation info
        if doc.metadata.source_type == SourceType.QURAN:
            citation["verse_key"] = doc.metadata.source_metadata.verse_key or doc.metadata.verse_key
            citation["text"] = doc.metadata.arabic_text or doc.text_content
        elif doc.metadata.source_type == SourceType.HADITH:
            citation["hadith_number"] = doc.metadata.source_metadata.hadith_number
            citation["authenticity"] = str(doc.metadata.source_metadata.authenticity_grade)
            citation["text"] = doc.text_content[:200] + "..."
        else:
            citation["text"] = doc.text_content[:150] + "..."
        
        citations.append(citation)
    
    return citations


# ============================================================================
# Utility Functions
# ============================================================================

def extract_top_sources(
    documents: List[DocumentChunk],
    n: int = 5,
    source_type: Optional[SourceType] = None
) -> List[DocumentChunk]:
    """
    Extract top N sources, optionally filtered by type.
    
    Args:
        documents: List of documents
        n: Number of sources to extract
        source_type: Optional source type filter
        
    Returns:
        List of top N documents
    """
    if source_type:
        filtered = [d for d in documents if d.metadata.source_type == source_type]
        return filtered[:n]
    
    return documents[:n]


def get_context_summary(documents: List[DocumentChunk]) -> Dict[str, Any]:
    """
    Get a summary of the context documents.
    
    Args:
        documents: List of documents
        
    Returns:
        Summary dictionary with stats
    """
    grouped = group_documents_by_source_type(documents)
    
    summary = {
        "total_sources": len(documents),
        "by_type": {
            str(source_type): len(docs) 
            for source_type, docs in grouped.items()
        },
        "has_quran": SourceType.QURAN in grouped,
        "has_sahih_hadith": any(
            d.metadata.source_type == SourceType.HADITH and 
            d.metadata.source_metadata.authenticity_grade == AuthenticityGrade.SAHIH
            for d in documents
        ),
    }
    
    return summary

