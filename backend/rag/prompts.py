"""
System prompts and templates for the Islamic Chatbot RAG system.

This module contains:
- Identity and methodology instructions
- Query classification prompts
- Query expansion prompts
- Response generation prompts
- Citation formatting templates
"""

from typing import Dict, List
from backend.core.models import QuestionType

# ============================================================================
# Identity & Methodology Prompt
# ============================================================================

SYSTEM_IDENTITY = """You are an Islamic knowledge assistant grounded in authentic Sunni sources.

**Core Principles:**
- Prioritize Quran, then Sahih Hadith, then scholarly consensus
- Every claim must cite specific sources
- Acknowledge scholarly differences respectfully
- Distinguish authentic from weak narrations
- Provide direct, clear answers with practical guidance when relevant

**Your responses should be:**
- Accurate and source-grounded
- Clear and accessible
- Properly cited with specific references (Surah:Verse, Hadith collection:number)
"""

# ============================================================================
# Query Complexity Analysis Prompt
# ============================================================================

QUERY_COMPLEXITY_PROMPT = """Analyze this query in the conversation context.

**Conversation History:**
{conversation_history}

**Current Query:** {user_query}

Classify as ONE of:
- simple_conversational: Greeting, thanks, acknowledgment (hi, hello, thank you, ok, got it)
- follow_up: References previous response (tell me more, what about X, can you clarify, elaborate on that)
- simple_factual: Basic factual question (when was prophet born, what is islam, who was Abu Bakr)
- complex: Requires deep source analysis (jurisprudence questions, theological discussions, detailed tafsir)

For follow_up queries, also determine:
- needs_new_retrieval: yes (if asking about new topic) or no (if expanding on current topic)

Return format:
complexity: [category]
needs_new_retrieval: [yes/no] (only if follow_up)"""

# ============================================================================
# Query Classification Prompt
# ============================================================================

QUERY_CLASSIFIER_PROMPT = """Classify the user question into one category:

**Categories:**
- general: General questions (default if unsure)
- fiqh: Jurisprudence, rulings, halal/haram, worship practices
- aqidah: Theology, beliefs, faith pillars
- tafsir: Quranic verse meanings and interpretations
- hadith: Prophetic sayings and traditions

**User Question:** {user_query}

Return ONLY one word: fiqh, aqidah, tafsir, hadith, or general"""

# ============================================================================
# Query Expansion Prompts
# ============================================================================

QUERY_EXPANSION_PROMPT = """Reformulate this question to improve retrieval from Islamic sources. Generate 2-3 alternative phrasings.

**Original Question:** {user_query}

**Question Type:** {question_type}

Make implicit Islamic concepts explicit, add relevant Arabic terminology, and use scholarly phrasing.

Example:
"What is charity?" →
1. What is charity (Sadaqah and Zakat) in Islam?
2. What do the Quran and Hadith say about charitable giving?

Generate 2-3 reformulated questions (one per line):"""

FIQH_QUERY_EXPANSION = """Reformulate this fiqh question to retrieve madhahib perspectives.

**Original Question:** {user_query}

Include terms like "ruling", "madhab", "Hanafi/Maliki/Shafi'i/Hanbali view", and practical application.

Generate 2-3 reformulated questions:"""

AQIDAH_QUERY_EXPANSION = """Reformulate this aqidah question for Quran and Hadith retrieval.

**Original Question:** {user_query}

Include terms like "belief", "faith", "iman", and focus on primary sources.

Generate 2-3 reformulated questions:"""

TAFSIR_QUERY_EXPANSION = """Reformulate this tafsir question for verse commentary retrieval.

**Original Question:** {user_query}

Include verse references if identifiable, terms like "tafsir", "interpretation", and scholar names (Ibn Kathir, Al-Tabari).

Generate 2-3 reformulated questions:"""

FOLLOW_UP_EXPANSION_PROMPT = """The user is asking a follow-up question about the previous topic.

**Previous Response:**
{previous_response}

**Previous Sources Used:**
{previous_sources}

**Follow-up Query:** {user_query}

Determine if this can be answered using the previous sources or requires new retrieval.

If the question can be answered by elaborating on the previous response, return: "use_existing"

If new retrieval is needed, provide 2-3 reformulated standalone queries (one per line):"""

# ============================================================================
# Context Ranking Prompt
# ============================================================================

CONTEXT_RANKING_INSTRUCTION = """Rank the following retrieved sources by relevance and authenticity for answering the user's question.

**Authenticity Hierarchy:**
1. Quran (highest authority)
2. Sahih Hadith (Bukhari, Muslim)
3. Hasan Hadith
4. Tafsir from recognized scholars
5. Fiqh rulings from established madhahib

**User Question:** {user_query}

**Sources:**
{sources}

Return the sources in ranked order (most relevant and authentic first)."""

# ============================================================================
# Response Generation Prompts
# ============================================================================

CONVERSATIONAL_RESPONSE_PROMPT = """You are an Islamic knowledge assistant.

**Conversation History:**
{conversation_history}

**User:** {user_query}

Respond naturally and concisely. For topics requiring Islamic sources, politely encourage them to ask a specific question so you can provide proper citations."""

RESPONSE_GENERATION_PROMPT = """Answer this question using the provided Islamic sources.

**Conversation History:**
{conversation_history}

**User Question:** {user_query}

**Retrieved Sources:**
{context}

**Instructions:**
1. Start with a clear, direct answer
2. Support with Quranic verses and authentic Hadith
3. Include relevant scholarly interpretations
4. Cite each source specifically (Surah:Verse, Hadith collection:number)
5. Acknowledge scholarly differences when present
6. Add practical guidance if relevant

Use markdown formatting naturally - headings, bold for key terms, blockquotes for verses/hadith, and bullet points where helpful."""

FIQH_GENERATION_PROMPT = """Answer this fiqh question using the provided sources, showing madhahib perspectives when available.

**Conversation History:**
{conversation_history}

**User Question:** {user_query}

**Retrieved Sources:**
{context}

**Instructions:**
1. State the general ruling clearly
2. Cite Quran and Hadith evidence
3. Present madhab views if sources provide them (Hanafi, Maliki, Shafi'i, Hanbali)
4. If madhahib agree, state consensus; if they differ, explain differences respectfully
5. Provide practical application guidance
6. Include specific citations

Use markdown naturally for organization."""

AQIDAH_GENERATION_PROMPT = """Answer this theological question using Quran and authentic Hadith.

**Conversation History:**
{conversation_history}

**User Question:** {user_query}

**Retrieved Sources:**
{context}

**Instructions:**
1. State the Islamic belief clearly
2. Cite relevant Quranic verses
3. Include authentic Hadith
4. Note scholarly consensus among Sunni scholars
5. Explain practical implications of this belief
6. Include specific citations

Use markdown naturally for organization."""

# ============================================================================
# Citation Formatting
# ============================================================================

CITATION_FORMAT_PROMPT = """Format the citations from the response into a structured list.

**Response with Citations:**
{response}

**Source Documents:**
{sources}

**Instructions:**
Extract and format citations as follows:

**Quranic Citations:**
- Surah Name (Surah:Verse) - "Verse text"

**Hadith Citations:**
- Collection Name, Book X, Hadith Y - "Hadith text (excerpt)"

**Scholarly Sources:**
- Author, Book Title - "Relevant excerpt"

Return formatted citations:"""

# ============================================================================
# Context Templates by Query Type
# ============================================================================

def get_context_template(question_type: QuestionType) -> str:
    """
    Get the appropriate context template based on question type.
    
    Args:
        question_type: The classified question type
        
    Returns:
        Formatted context template string
    """
    templates = {
        QuestionType.FIQH: """
**PRIMARY EVIDENCE:**
{quran_verses}

{sahih_hadiths}

**JURISPRUDENTIAL VIEWS:**
{fiqh_rulings}

**SCHOLARLY INTERPRETATION:**
{tafsir_commentary}
""",
        QuestionType.AQIDAH: """
**PRIMARY EVIDENCE:**
{quran_verses}

{sahih_hadiths}

**SCHOLARLY INTERPRETATION:**
{tafsir_commentary}

**THEOLOGICAL CONTEXT:**
{aqidah_sources}
""",
        QuestionType.TAFSIR: """
**QURANIC VERSE:**
{quran_verses}

**CLASSICAL TAFSIR:**
{tafsir_commentary}

**RELATED HADITH:**
{sahih_hadiths}

**SCHOLARLY NOTES:**
{additional_context}
""",
        QuestionType.HADITH: """
**HADITH TEXT:**
{sahih_hadiths}

**QURANIC CONTEXT:**
{quran_verses}

**SCHOLARLY EXPLANATION:**
{tafsir_commentary}

**RELATED TRADITIONS:**
{related_hadiths}
""",
        QuestionType.GENERAL: """
**QURANIC FOUNDATION:**
{quran_verses}

**PROPHETIC GUIDANCE:**
{sahih_hadiths}

**SCHOLARLY INTERPRETATION:**
{tafsir_commentary}

**PRACTICAL APPLICATION:**
{practical_guidance}
"""
    }
    
    return templates.get(question_type, templates[QuestionType.GENERAL])


def get_generation_prompt(question_type: QuestionType) -> str:
    """
    Get the appropriate generation prompt based on question type.
    
    Args:
        question_type: The classified question type
        
    Returns:
        Generation prompt string
    """
    prompts = {
        QuestionType.FIQH: FIQH_GENERATION_PROMPT,
        QuestionType.AQIDAH: AQIDAH_GENERATION_PROMPT,
        QuestionType.TAFSIR: RESPONSE_GENERATION_PROMPT,
        QuestionType.HADITH: RESPONSE_GENERATION_PROMPT,
        QuestionType.GENERAL: RESPONSE_GENERATION_PROMPT,
    }
    
    return prompts.get(question_type, RESPONSE_GENERATION_PROMPT)


def get_expansion_prompt(question_type: QuestionType) -> str:
    """
    Get the appropriate query expansion prompt based on question type.
    
    Args:
        question_type: The classified question type
        
    Returns:
        Query expansion prompt string
    """
    prompts = {
        QuestionType.FIQH: FIQH_QUERY_EXPANSION,
        QuestionType.AQIDAH: AQIDAH_QUERY_EXPANSION,
        QuestionType.TAFSIR: TAFSIR_QUERY_EXPANSION,
        QuestionType.HADITH: QUERY_EXPANSION_PROMPT,
        QuestionType.GENERAL: QUERY_EXPANSION_PROMPT,
    }
    
    return prompts.get(question_type, QUERY_EXPANSION_PROMPT)


# ============================================================================
# Helper Functions
# ============================================================================

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided arguments.
    
    Args:
        template: Prompt template string with {placeholders}
        **kwargs: Values to fill in the placeholders
        
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)


def create_system_message(include_identity: bool = True) -> Dict[str, str]:
    """
    Create a system message for the LLM.
    
    Args:
        include_identity: Whether to include the full identity prompt
        
    Returns:
        Dict with role and content
    """
    return {
        "role": "system",
        "content": SYSTEM_IDENTITY if include_identity else "You are a helpful Islamic knowledge assistant."
    }

