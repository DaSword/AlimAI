"""
Test fixtures and sample data for RAG node testing.

This module provides:
- Sample document chunks (Quran, Hadith, Tafsir, Fiqh)
- Sample conversation histories
- Mock LLM responses
- Response validators
- Helper functions for creating test states
"""

from typing import List, Dict, Any
import re
from backend.core.models import (
    DocumentChunk,
    QdrantPayload,
    SourceType,
    AuthenticityGrade,
    Madhab,
    QuestionType,
)


# ============================================================================
# Sample Document Chunks
# ============================================================================

def create_sample_quran_verse(
    surah_number: int = 1,
    verse_number: int = 1,
    arabic_text: str = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    english_text: str = "In the name of Allah, the Entirely Merciful, the Especially Merciful.",
    score: float = 0.95,
) -> DocumentChunk:
    """Create a sample Quran verse document chunk."""
    return DocumentChunk(
        id=f"quran_{surah_number}_{verse_number}",
        text_content=f"{arabic_text}\n\n{english_text}",
        metadata=QdrantPayload(
            source_type=SourceType.QURAN,
            book_title="The Noble Quran",
            book_title_arabic="القرآن الكريم",
            author="Allah (SWT)",
            text_content=english_text,
            arabic_text=arabic_text,
            english_text=english_text,
            verse_key=f"{surah_number}:{verse_number}",
            source_metadata={
                "surah_number": surah_number,
                "verse_number": verse_number,
                "verse_key": f"{surah_number}:{verse_number}",
                "surah_name": "Al-Fatiha" if surah_number == 1 else f"Surah {surah_number}",
            },
            topic_tags=["quran", "revelation"],
        ),
        score=score,
    )


def create_sample_hadith(
    collection: str = "Sahih Bukhari",
    hadith_number: int = 1,
    text: str = "The reward of deeds depends upon the intentions...",
    authenticity: AuthenticityGrade = AuthenticityGrade.SAHIH,
    score: float = 0.90,
) -> DocumentChunk:
    """Create a sample Hadith document chunk."""
    return DocumentChunk(
        id=f"hadith_{collection}_{hadith_number}",
        text_content=text,
        metadata=QdrantPayload(
            source_type=SourceType.HADITH,
            book_title=collection,
            author="Prophet Muhammad (ﷺ)",
            text_content=text,
            source_metadata={
                "hadith_number": hadith_number,
                "book_name": collection,
                "authenticity_grade": authenticity,
            },
            topic_tags=["hadith", "prophetic tradition"],
        ),
        score=score,
    )


def create_sample_tafsir(
    verse_key: str = "1:1",
    tafsir_source: str = "Ibn Kathir",
    text: str = "This verse is the opening of the Quran...",
    score: float = 0.85,
) -> DocumentChunk:
    """Create a sample Tafsir document chunk."""
    return DocumentChunk(
        id=f"tafsir_{verse_key}_{tafsir_source}",
        text_content=text,
        metadata=QdrantPayload(
            source_type=SourceType.TAFSIR,
            book_title=f"Tafsir {tafsir_source}",
            author=tafsir_source,
            text_content=text,
            verse_key=verse_key,
            source_metadata={
                "verse_key": verse_key,
                "tafsir_source": tafsir_source,
            },
            topic_tags=["tafsir", "exegesis"],
        ),
        score=score,
    )


def create_sample_fiqh_ruling(
    madhab: Madhab = Madhab.HANAFI,
    ruling_category: str = "Prayer",
    text: str = "It is obligatory to perform prayer five times daily...",
    score: float = 0.88,
) -> DocumentChunk:
    """Create a sample Fiqh ruling document chunk."""
    return DocumentChunk(
        id=f"fiqh_{madhab.value}_{ruling_category}",
        text_content=text,
        metadata=QdrantPayload(
            source_type=SourceType.FIQH,
            book_title=f"{madhab.value.capitalize()} Fiqh Rulings",
            author=f"{madhab.value.capitalize()} Scholars",
            text_content=text,
            source_metadata={
                "madhab": madhab,
                "ruling_category": ruling_category,
            },
            topic_tags=["fiqh", "jurisprudence", madhab.value],
        ),
        score=score,
    )


def get_sample_documents() -> List[DocumentChunk]:
    """Get a collection of sample documents for testing."""
    return [
        create_sample_quran_verse(
            surah_number=2,
            verse_number=183,
            arabic_text="يَا أَيُّهَا الَّذِينَ آمَنُوا كُتِبَ عَلَيْكُمُ الصِّيَامُ",
            english_text="O you who have believed, decreed upon you is fasting as it was decreed upon those before you that you may become righteous.",
            score=0.95,
        ),
        create_sample_hadith(
            collection="Sahih Bukhari",
            hadith_number=1903,
            text="Whoever observes fasts during the month of Ramadan out of sincere faith, and hoping to attain Allah's rewards, then all his past sins will be forgiven.",
            authenticity=AuthenticityGrade.SAHIH,
            score=0.92,
        ),
        create_sample_tafsir(
            verse_key="2:183",
            tafsir_source="Ibn Kathir",
            text="Allah ordained fasting for Muslims as He had ordained it for previous nations, so that they may attain taqwa (consciousness of Allah).",
            score=0.88,
        ),
        create_sample_fiqh_ruling(
            madhab=Madhab.HANAFI,
            ruling_category="Fasting",
            text="According to the Hanafi school, the intention (niyyah) for fasting in Ramadan must be made before Fajr.",
            score=0.85,
        ),
        create_sample_fiqh_ruling(
            madhab=Madhab.SHAFI,
            ruling_category="Fasting",
            text="The Shafi'i school holds that the intention for obligatory fasting can be made at any point during the night before dawn.",
            score=0.84,
        ),
    ]


# ============================================================================
# Sample Conversation Histories
# ============================================================================

def create_empty_conversation() -> List[Dict[str, Any]]:
    """Create an empty conversation history."""
    return []


def create_simple_conversation() -> List[Dict[str, Any]]:
    """Create a simple 2-turn conversation."""
    return [
        {"role": "user", "content": "What is fasting in Islam?"},
        {
            "role": "assistant",
            "content": "Fasting (Sawm) in Islam is one of the Five Pillars and involves abstaining from food, drink, and intimate relations from dawn until sunset. The most important fast is during the month of Ramadan, which is obligatory for all adult Muslims.",
        },
    ]


def create_multi_turn_conversation() -> List[Dict[str, Any]]:
    """Create a multi-turn conversation with follow-ups."""
    return [
        {"role": "user", "content": "What is charity in Islam?"},
        {
            "role": "assistant",
            "content": "Charity in Islam includes both obligatory Zakat (2.5% of wealth) and voluntary Sadaqah. The Quran emphasizes charity extensively.",
        },
        {"role": "user", "content": "Tell me more about Zakat"},
        {
            "role": "assistant",
            "content": "Zakat is the obligatory almsgiving, one of the Five Pillars of Islam. It is 2.5% of one's savings and is given to specific categories of recipients mentioned in the Quran.",
        },
        {"role": "user", "content": "Who can receive Zakat?"},
    ]


def create_conversational_history() -> List[Dict[str, Any]]:
    """Create a conversational exchange with greetings."""
    return [
        {"role": "user", "content": "Assalamu alaikum"},
        {"role": "assistant", "content": "Wa alaikum assalam! How can I help you with Islamic knowledge today?"},
    ]


# ============================================================================
# Mock LLM Responses
# ============================================================================

class MockLLMResponse:
    """Mock LLM response object."""
    
    def __init__(self, content: str):
        self.content = content
        self.message = self
    
    def __str__(self):
        return self.content


def get_mock_complexity_response(complexity: str, needs_new_retrieval: bool = True) -> str:
    """Get a mock response for complexity analysis."""
    retrieval_line = f"\nneeds_new_retrieval: {'yes' if needs_new_retrieval else 'no'}" if complexity == "follow_up" else ""
    return f"complexity: {complexity}{retrieval_line}"


def get_mock_classification_response(question_type: QuestionType) -> str:
    """Get a mock response for query classification."""
    return question_type.value


def get_mock_expansion_response(original_query: str, num_expansions: int = 2) -> str:
    """Get a mock response for query expansion."""
    expansions = [
        f"Expanded version 1: {original_query} in Islamic perspective",
        f"Expanded version 2: What do Quran and Hadith say about {original_query}",
    ]
    return "\n".join(expansions[:num_expansions])


def get_mock_generation_response(query: str, has_context: bool = True) -> str:
    """Get a mock response for answer generation."""
    if has_context:
        return f"""# Answer to: {query}

Based on Islamic sources, here is the answer with proper citations.

**Key Points:**
- Point 1 from Quran
- Point 2 from Hadith

> "This is a quoted verse or hadith"

As mentioned in Surah 2:183, this is explained clearly."""
    else:
        return f"I don't have specific sources for '{query}', but generally in Islam this topic relates to..."


def get_mock_conversational_response(query: str) -> str:
    """Get a mock response for conversational queries."""
    greetings = ["hello", "hi", "assalam", "salam"]
    thanks = ["thank", "thanks", "jazak"]
    
    query_lower = query.lower()
    if any(g in query_lower for g in greetings):
        return "Wa alaikum assalam! How can I assist you with Islamic knowledge today?"
    elif any(t in query_lower for t in thanks):
        return "You're welcome! Feel free to ask if you have more questions."
    else:
        return "I'm here to help with questions about Islam. What would you like to know?"


# ============================================================================
# Mock Retriever
# ============================================================================

class MockRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, documents: List[DocumentChunk] = None):
        self.documents = documents or get_sample_documents()
    
    def retrieve_for_question_type(
        self,
        query: str,
        question_type: QuestionType,
        top_k: int = 10,
        madhab_preference: str = None,
    ) -> List[DocumentChunk]:
        """Mock retrieval that returns sample documents."""
        # Filter by question type if specific
        if question_type == QuestionType.FIQH:
            docs = [d for d in self.documents if d.metadata.source_type == SourceType.FIQH]
        elif question_type == QuestionType.HADITH:
            docs = [d for d in self.documents if d.metadata.source_type == SourceType.HADITH]
        elif question_type == QuestionType.TAFSIR:
            docs = [d for d in self.documents if d.metadata.source_type == SourceType.TAFSIR]
        else:
            docs = self.documents
        
        # Filter by madhab if specified
        if madhab_preference:
            madhab_docs = [
                d for d in docs 
                if d.metadata.source_metadata.get("madhab") == madhab_preference
            ]
            if madhab_docs:
                docs = madhab_docs
        
        return docs[:top_k]


# ============================================================================
# Response Validators
# ============================================================================

def validate_islamic_style(response: str) -> bool:
    """
    Validate that the response uses Islamic terminology and respectful tone.
    
    Checks for:
    - Respectful references (Allah, Prophet Muhammad)
    - Islamic terminology presence
    - No inappropriate language
    """
    # Check for basic Islamic terminology
    islamic_terms = [
        "allah", "islam", "prophet", "quran", "hadith", 
        "surah", "verse", "muslim", "sawm", "salah", "zakat"
    ]
    response_lower = response.lower()
    
    # Should contain at least some Islamic terminology
    has_islamic_terms = any(term in response_lower for term in islamic_terms)
    
    # Check for respectful tone (no inappropriate words)
    # This is a simple check - can be expanded
    inappropriate_terms = ["stupid", "dumb", "idiot"]
    has_inappropriate = any(term in response_lower for term in inappropriate_terms)
    
    return has_islamic_terms and not has_inappropriate


def validate_citation_format(response: str) -> bool:
    """
    Validate that citations are properly formatted.
    
    Checks for:
    - Surah:Verse format (e.g., "2:183", "Al-Baqarah 2:183")
    - Hadith references (e.g., "Sahih Bukhari", "Hadith 1903")
    """
    # Check for Quran citations (Surah:Verse)
    quran_pattern = r'\d+:\d+'
    has_quran_citation = re.search(quran_pattern, response) is not None
    
    # Check for Hadith references
    hadith_collections = ["bukhari", "muslim", "tirmidhi", "abu dawud", "nasai", "ibn majah"]
    response_lower = response.lower()
    has_hadith_reference = any(collection in response_lower for collection in hadith_collections)
    
    # At least one type of citation should be present if response is substantive
    return has_quran_citation or has_hadith_reference or len(response) < 50


def validate_markdown_usage(response: str) -> bool:
    """
    Validate that the response uses markdown formatting.
    
    Checks for:
    - Headings (# or ##)
    - Bold (**text**)
    - Blockquotes (> text)
    - Lists (- or 1.)
    """
    markdown_patterns = [
        r'^#{1,6}\s',  # Headings
        r'\*\*[^*]+\*\*',  # Bold
        r'^>\s',  # Blockquotes
        r'^[-*]\s',  # Unordered lists
        r'^\d+\.\s',  # Ordered lists
    ]
    
    # Check if any markdown pattern is present
    for pattern in markdown_patterns:
        if re.search(pattern, response, re.MULTILINE):
            return True
    
    # For very short responses, markdown might not be necessary
    return len(response) < 100


def validate_response_structure(response: str) -> bool:
    """
    Validate that the response has a clear structure.
    
    Checks for:
    - Multiple paragraphs or sections
    - Clear organization
    """
    # Check for multiple paragraphs (separated by double newlines)
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    has_structure = len(paragraphs) >= 2 or len(response) < 100
    
    return has_structure


def validate_response_quality(response: str) -> Dict[str, bool]:
    """
    Run all response quality validators.
    
    Returns:
        Dict with validation results for each check
    """
    return {
        "islamic_style": validate_islamic_style(response),
        "citation_format": validate_citation_format(response),
        "markdown_usage": validate_markdown_usage(response),
        "response_structure": validate_response_structure(response),
    }


# ============================================================================
# Helper Functions for Test States
# ============================================================================

def create_test_state(
    user_query: str = "What is fasting in Islam?",
    messages: List[Dict[str, Any]] = None,
    query_complexity: str = None,
    question_type: str = None,
    retrieved_docs: List[DocumentChunk] = None,
    ranked_docs: List[DocumentChunk] = None,
    response: str = None,
    max_sources: int = 10,
    score_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Create a test state dictionary for RAG workflow."""
    state = {
        "user_query": user_query,
        "max_sources": max_sources,
        "score_threshold": score_threshold,
    }
    
    if messages is not None:
        state["messages"] = messages
    
    if query_complexity is not None:
        state["query_complexity"] = query_complexity
    
    if question_type is not None:
        state["question_type"] = question_type
    
    if retrieved_docs is not None:
        # Convert to dict format
        state["retrieved_docs"] = [
            {
                "id": doc.id,
                "text_content": doc.text_content,
                "metadata": doc.metadata.model_dump(),
                "score": doc.score,
            }
            for doc in retrieved_docs
        ]
    
    if ranked_docs is not None:
        # Convert to dict format
        state["ranked_docs"] = [
            {
                "id": doc.id,
                "text_content": doc.text_content,
                "metadata": doc.metadata.model_dump(),
                "score": doc.score,
            }
            for doc in ranked_docs
        ]
    
    if response is not None:
        state["response"] = response
    
    return state


def create_test_state_with_messages(
    user_query: str,
    conversation_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a test state with conversation history."""
    return create_test_state(
        user_query=user_query,
        messages=conversation_history + [{"role": "user", "content": user_query}],
    )

