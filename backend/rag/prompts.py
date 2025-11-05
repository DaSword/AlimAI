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

SYSTEM_IDENTITY = """You are an Islamic knowledge assistant grounded in authentic Sunni sources. Your purpose is to provide accurate, source-backed answers to questions about Islam.

**Core Principles:**

1. **Source Priority:**
   - Always prioritize the Quran as the primary source
   - Follow with authentic Hadith (Sahih al-Bukhari, Sahih Muslim)
   - Include scholarly interpretations from recognized authorities
   - Present balanced views from the four madhahib (schools of jurisprudence) when relevant

2. **Methodology:**
   - Every claim must be backed by specific sources with citations
   - Never contradict clear Quranic or authentic Hadith teachings
   - Acknowledge scholarly differences respectfully
   - Distinguish between authentic and weak narrations
   - Avoid sectarian polemics

3. **Response Framework:**
   - Provide direct, clear answers
   - Support with Quranic verses and authentic Hadith
   - Include relevant scholarly interpretations
   - Offer practical guidance when applicable

4. **Islamic Perspective:**
   - Frame general questions within an Islamic context when appropriate
   - Connect universal concepts to Islamic teachings
   - Maintain respect for Islamic tradition and scholarship

**Your responses should be:**
- Accurate and source-grounded
- Clear and accessible
- Balanced and respectful
- Properly cited with specific references
"""

# ============================================================================
# Query Classification Prompt
# ============================================================================

QUERY_CLASSIFIER_PROMPT = """Classify the following user question into one of these categories:

**Categories:**
- **general**: General questions about Islam, Islamic history, ethics, or broad topics (If unsure, classify as general)
- **fiqh**: Questions about Islamic jurisprudence, rulings, halal/haram, how to perform acts of worship
- **aqidah**: Questions about Islamic theology, beliefs, pillars of faith, God's attributes
- **tafsir**: Questions about Quranic verses, their meanings, or interpretations
- **hadith**: Questions asking about specific prophetic sayings or traditions

**Examples:**
- "How do I pray Witr?" → fiqh
- "What are the attributes of Allah?" → aqidah
- "What does Ayat al-Kursi mean?" → tafsir
- "What did the Prophet say about intentions?" → hadith
- "What is the purpose of life in Islam?" → general

**User Question:** {user_query}

**Instructions:**
- If unsure, classify as general
- If the question is about a specific topic, classify it into the most appropriate category
- Be VERY CONCISE, only return one word from the categories above.

**Classification:** Return ONLY one word from: fiqh, aqidah, tafsir, hadith, general. If unsure, classify as general."""

# ============================================================================
# Query Expansion Prompts
# ============================================================================

QUERY_EXPANSION_PROMPT = """Reformulate the following question to make it more specific and Islam-centric. Generate 2-3 alternative phrasings that would help retrieve relevant Islamic sources.

**Original Question:** {user_query}

**Question Type:** {question_type}

**Instructions:**
- Make implicit Islamic concepts explicit
- Add relevant Arabic terminology
- Include alternative phrasings that scholars might use
- Keep questions focused and specific

**Example:**
Original: "What is charity?"
Reformulated:
1. "What is charity (Sadaqah and Zakat) in Islam?"
2. "What are the Islamic teachings on charitable giving?"
3. "What do the Quran and Hadith say about charity and helping the poor?"

Generate 2-3 reformulated questions as a list (one question per line):"""

FIQH_QUERY_EXPANSION = """Reformulate this fiqh question to retrieve rulings from multiple madhahib (schools of jurisprudence).

**Original Question:** {user_query}

**Instructions:**
- Include terms like "ruling", "madhab", "jurisprudence"
- Consider different schools: Hanafi, Maliki, Shafi'i, Hanbali
- Focus on practical application

Generate 2-3 reformulated questions as a list:"""

AQIDAH_QUERY_EXPANSION = """Reformulate this aqidah (theology) question to prioritize Quranic and authentic Hadith sources.

**Original Question:** {user_query}

**Instructions:**
- Include terms like "belief", "faith", "iman"
- Focus on Quran and Sahih Hadith
- Consider classical theological terminology

Generate 2-3 reformulated questions as a list:"""

TAFSIR_QUERY_EXPANSION = """Reformulate this tafsir question to retrieve verse-specific commentary.

**Original Question:** {user_query}

**Instructions:**
- Include verse references if identifiable
- Use terms like "tafsir", "interpretation", "meaning"
- Include scholar names when relevant (Ibn Kathir, Al-Tabari, etc.)

Generate 2-3 reformulated questions as a list:"""

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

RESPONSE_GENERATION_PROMPT = """Based on the authentic Islamic sources provided, generate a comprehensive answer to the user's question.

**User Question:** {user_query}

**Retrieved Sources:**
{context}

**Instructions:**
1. **Answer directly** - Start with a clear, direct answer
2. **Provide evidence** - Support with Quranic verses and authentic Hadith
3. **Include scholarship** - Add relevant interpretations from recognized scholars
4. **Cite specifically** - Reference each source with precise citations (Surah:Verse, Hadith book:number)
5. **Acknowledge differences** - When applicable, present multiple scholarly views
6. **Be practical** - Include practical guidance when relevant

**Response Format (Use Markdown):**
- Start with a direct answer in a paragraph
- Use **bold** for emphasis on key Islamic terms
- Use proper headings (##, ###) to organize different sections
- Use bullet points (-) or numbered lists (1., 2., 3.) for multiple points
- Use > blockquotes for Quranic verses and Hadith quotations
- Format citations clearly at the end

**Example Structure:**
## Direct Answer
[Clear, concise answer]

## Quranic Guidance
> "[Quranic verse text]" (Surah Name X:Y)

## Prophetic Guidance
> [Hadith text] (Sahih Bukhari, Hadith #)

## Scholarly Interpretation
- Point 1
- Point 2

## Practical Guidance
[Practical advice]

Generate your response using proper markdown formatting:"""

FIQH_GENERATION_PROMPT = """Based on the authentic Islamic sources provided, generate a comprehensive fiqh answer showing perspectives from multiple madhahib when applicable.

**User Question:** {user_query}

**Retrieved Sources from Madhahib:**
{context}

**Instructions:**
1. **Direct Answer** - Provide the general ruling
2. **Evidence** - Cite Quran and Hadith supporting the ruling
3. **Madhab Views** - Present views from different schools:
   - Hanafi:
   - Maliki:
   - Shafi'i:
   - Hanbali:
4. **Practical Guidance** - Explain how to apply the ruling
5. **Citations** - Include specific source references

**Response Format (Use Markdown):**
- Use **bold** for key Islamic terms and rulings
- Use ## headings to organize sections
- Use bullet points or numbered lists for madhab differences
- Use > blockquotes for Quranic verses and Hadith
- Format madhab views clearly

**Example Structure:**
## Ruling Summary
[Direct answer with ruling]

## Evidence from Primary Sources
> [Quranic verse or Hadith]

## Views of the Four Madhahib
- **Hanafi:** [View]
- **Maliki:** [View]
- **Shafi'i:** [View]
- **Hanbali:** [View]

## Practical Application
[Step-by-step guidance]

**Note:** If madhahib agree, state the consensus. If they differ, explain the differences respectfully.

Generate your response using proper markdown formatting:"""

AQIDAH_GENERATION_PROMPT = """Based on the Quran and authentic Hadith, provide a clear answer about this matter of Islamic belief.

**User Question:** {user_query}

**Retrieved Sources:**
{context}

**Instructions:**
1. **Core Teaching** - State the Islamic belief clearly
2. **Quranic Foundation** - Cite relevant verses
3. **Prophetic Guidance** - Include authentic Hadith
4. **Scholarly Consensus** - Note consensus among Sunni scholars
5. **Practical Implications** - Explain how this belief affects practice

**Response Format (Use Markdown):**
- Use **bold** for key theological concepts
- Use ## headings to organize sections
- Use > blockquotes for Quranic verses and Hadith
- Use bullet points or numbered lists for multiple points
- Clearly separate foundational evidence from practical implications

**Example Structure:**
## Core Belief
[Clear statement of the Islamic belief]

## Quranic Foundation
> "[Quranic verse]" (Surah Name X:Y)

## Prophetic Guidance
> [Hadith text] (Collection, Hadith #)

## Scholarly Consensus
[Explanation of scholarly agreement]

## Practical Implications
- How this belief affects daily practice
- How it shapes a Muslim's worldview

Generate your response using proper markdown formatting:"""

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

