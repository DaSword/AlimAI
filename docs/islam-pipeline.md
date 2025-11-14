# Islamic Chatbot RAG Implementation Plan

---

## Phase 1: Book Selection & Prioritization

### Tier 1 - Essential Core (Highest Priority)

**Primary Sources:**

1. **Quran** (already ingested) - Foundation of all Islamic knowledge
2. **Sahih al-Bukhari** - Most authentic Hadith collection
3. **Sahih Muslim** - Second most authentic Hadith collection

**Commentary:**

4. **Tafsir Ibn Kathir** (or Al-Tabari's Jami' al-bayan) - Comprehensive Quranic exegesis
5. **Riyad al-Salihin** - Practical ethical Hadith compilation

**Biography:**

6. **Al-Sirah al-Nabawiyyah** (Ibn Hisham) - Prophet's biography

### Tier 2 - Balanced Jurisprudence (Medium Priority)

**Fiqh across Madhahib:**

7. **Al-Hidayah** (Hanafi) - Standard Hanafi legal code
8. **Al-Muwatta'** (Maliki) - Imam Malik's compilation
9. **Matn al-Ghayah wa al-Taqrib** (Shafi'i) - Core Shafi'i text
10. **'Umdat al-Fiqh** (Hanbali) - Core Hanbali manual

**Additional Hadith:**

11. **Sunan Abi Dawud** - Legal rulings focus
12. **Jami' al-Tirmidhi** - Hadith strength classifications

### Tier 3 - Depth & Methodology (Lower Priority)

**Sciences & Methodology:**

13. **Al-Itqan fi Ulum al-Qur'an** - Quranic sciences
14. **Muqaddimat Ibn al-Salah** - Hadith methodology
15. **Al-Risala** (Al-Shafi'i) - Usul al-Fiqh foundation

**Theology & Ethics:**

16. **Al-Aqidah al-Wasitiyyah** - Athari creed
17. **Ihya Ulum al-Din** - Spiritual purification

**Why this selection:** Provides comprehensive coverage across Quran, Hadith, Fiqh (all 4 madhahib), Aqidah, Seerah, and Methodology while maintaining Sunni orthodoxy and allowing nuanced, multi-perspective answers.

## Phase 2: Project Structure & Organization

```
alimai/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env
│
├── data/                           # Source texts and datasets
│   ├── quran.json                 # Existing Quran data
│   ├── islamic_knowledge.json     # Existing collection export
│   ├── hadith/
│   │   ├── sahih_bukhari.json
│   │   ├── sahih_muslim.json
│   │   ├── sunan_abu_dawud.json
│   │   └── jami_tirmidhi.json
│   ├── tafsir/
│   │   └── tafsir_ibn_kathir.json
│   ├── seerah/
│   │   └── ibn_hisham_seerah.json
│   ├── fiqh/
│   │   ├── hanafi/
│   │   ├── maliki/
│   │   ├── shafi/
│   │   └── hanbali/
│   └── aqidah/
│       └── aqidah_wasitiyyah.json
│
├── backend/                        # Modular backend with Python packages
│   ├── __init__.py                 # Makes backend a package
│   │
│   ├── core/                       # ✅ Core configuration & models
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── models.py               # Pydantic schemas (state, API, documents)
│   │   └── utils.py                # General utilities & logging
│   │
│   ├── llama/                      # ✅ LlamaIndex/Ollama integration
│   │   ├── __init__.py
│   │   ├── llama_config.py         # LlamaIndex Settings and Ollama configuration
│   │   ├── embeddings_service.py   # Wrapper around LlamaIndex OllamaEmbedding
│   │   ├── llm_service.py          # Wrapper around LlamaIndex Ollama LLM
│   │   └── reranker_service.py     # LLM-based reranker
│   │
│   ├── vectordb/                   # ✅ Vector database operations
│   │   ├── __init__.py
│   │   └── qdrant_manager.py       # Qdrant + LlamaIndex VectorStore integration
│   │
│   ├── ingestion/                  # ✅ Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── parsers.py              # LlamaIndex NodeParsers (Quran, Hadith, Tafsir, Fiqh, Seerah)
│   │   ├── chunking.py             # Text chunking utilities
│   │   ├── ingestion.py            # LlamaIndex IngestionPipeline orchestration
│   │   └── migrate_data.py         # Data migration utility
│   │
│   ├── rag/                        # Stage 5: RAG components
│   │   ├── __init__.py
│   │   ├── rag_graph.py            # LangGraph StateGraph workflow definition
│   │   ├── rag_nodes.py            # Individual LangGraph workflow nodes
│   │   ├── retrieval.py            # LlamaIndex query engine wrapper
│   │   ├── context_formatter.py    # Format retrieved context for RAG
│   │   └── prompts.py              # System prompts and templates
│   │
│   ├── api/                        # Stage 5: Server & API
│   │   ├── __init__.py
│   │   ├── server.py               # LangGraph Server setup and configuration
│   │   └── admin/
│   │       ├── __init__.py
│   │       ├── ingestion_api.py    # Admin endpoints for ingestion
│   │       ├── collection_api.py   # Admin endpoints for collection management
│   │       └── models_api.py       # Admin endpoints for model management
│   │
│   └── tests/                      # ✅ Test suite
│       ├── __init__.py
│       ├── test_stage4.py          # Stage 4 ingestion tests
│       └── test_imports.py         # Import verification tests
│
├── langgraph.json                  # LangGraph Server configuration
│
├── frontend/                       # Vite + React frontend
│   ├── src/
│   │   ├── App.jsx                 # Main app component with routing
│   │   ├── main.jsx                # Entry point
│   │   ├── pages/
│   │   │   ├── Chat.jsx            # Main chat interface
│   │   │   └── Admin.jsx           # Admin dashboard
│   │   ├── components/
│   │   │   ├── ChatBox.jsx         # Reusable chat input/output
│   │   │   ├── MessageList.jsx     # Display message history
│   │   │   ├── SourceCitation.jsx  # Display retrieved sources with references
│   │   │   └── admin/
│   │   │       ├── IngestionPanel.jsx      # Upload and ingest texts
│   │   │       ├── CollectionManager.jsx   # View/manage Qdrant collections
│   │   │       └── ModelStatus.jsx         # Check Ollama models status
│   │   └── api/
│   │       └── client.js           # API client for backend communication
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
│
└── logs/                           # Application logs
    └── .gitkeep
```

**Key Organizational Principles:**

1. **LangGraph Server:** Native API server with built-in streaming, state persistence, and thread management
2. **Modular Architecture** 
3. **Import Style:** `from backend.core.config import Config`, `from backend.vectordb.qdrant_manager import QdrantManager`
4. **Separation of Concerns:** Frontend (React), Backend (LangGraph Server), RAG (LlamaIndex + LangGraph), Data (JSON files), Infrastructure (Docker)
5. **LlamaIndex Integration:** All indexing, retrieval, and query operations use LlamaIndex
6. **LangGraph Orchestration:** Complex RAG workflow managed by LangGraph StateGraph with native API endpoints
7. **LM Studio Integration:** All models (embedding, reranker, chat) hosted through LM Studio, accessed via LlamaIndex
8. **Admin UI:** All ingestion, migration, and management functionality accessible through React admin panel (no standalone scripts)
9. **Existing Files Migration:**
   - `embeddings_manager.py` → `backend/llama/embeddings_service.py` (wrapper around LlamaIndex)
   - `qdrant_manager.py` → `backend/vectordb/qdrant_manager.py` (updated for LlamaIndex integration)
   - `quran_chunker.py` → `backend/ingestion/chunking.py` (converted to LlamaIndex NodeParsers)
   - All parsers in `backend/ingestion/parsers.py` (QuranNodeParser, HadithNodeParser, etc.)
10. **Configuration:** 
    - `backend/core/config.py` - General configuration
    - `backend/llama/llama_config.py` - LlamaIndex settings
    - `langgraph.json` - LangGraph Server configuration

## Phase 3: Data Ingestion Architecture (LlamaIndex)

**See `RAG-plan.md` for detailed LlamaIndex implementation patterns and examples.**

### 3.1 LlamaIndex Document & Node Structure

Uses LlamaIndex's Document → Node transformation pipeline:

- **Documents:** Raw text loaded from JSON files
- **Nodes:** Chunked text with metadata, created by custom NodeParsers
- **Embeddings:** Generated via LlamaIndex's Ollama integration
- **Vector Store:** Qdrant integration via `llama-index-vector-stores-qdrant`

### 3.2 Universal Text Chunker (`backend/chunking.py`)

Custom LlamaIndex NodeParsers for each text type (extends `NodeParser` base class):

- **HadithNodeParser:** By individual Hadith with chain of narration (isnad) + text (matn) + book/chapter metadata
- **TafsirNodeParser:** By verse commentary sections
- **FiqhNodeParser:** By legal topic/ruling with hierarchical context
- **SeerahNodeParser:** By chronological events/sections
- **QuranNodeParser:** By verse with surah context (migrated from `quran_chunker.py`)

Key features:

- Preserve hierarchical structure (Book → Chapter → Section → Hadith/Ruling)
- Include cross-references and narrator chains in metadata
- Maintain Arabic + English pairs
- Smart chunk sizing (500-2000 chars based on text type)
- Generate structured metadata for LlamaIndex Documents

### 3.3 Design Clean Qdrant Schema

Use a **compact core + flexible metadata** approach to avoid sparse fields:

```python
payload = {
    # ===== CORE FIELDS (all sources have these) =====
    'source_type': 'quran|hadith|tafsir|fiqh|seerah|aqidah|usul',
    'book_title': 'Sahih al-Bukhari',
    'book_title_arabic': 'صحيح البخاري',
    'author': 'Muhammad ibn Isma\'il al-Bukhari',
    'text_content': '...',  # The main searchable text
    'arabic_text': '...',   # Arabic original (if applicable)
    'english_text': '...',  # English translation (if applicable)
    'topic_tags': ['prayer', 'purification', 'charity'],  # For all types
    
    # ===== SOURCE-SPECIFIC METADATA (nested, only populated as needed) =====
    'source_metadata': {
        # For Hadith:
        'hadith_number': 8,
        'book_name': 'Book of Faith',
        'chapter_number': 2,
        'authenticity_grade': 'sahih',
        'narrator_chain': '...',
        
        # For Fiqh:
        'madhab': 'hanafi',
        'ruling_category': 'prayer',
        
        # For Quran/Tafsir:
        'surah_number': 2,
        'verse_number': 183,
        'verse_key': '2:183',
        'surah_name': 'Al-Baqarah',
        'chunk_type': 'verse|tafsir',
        'chunk_index': 0,
        'tafsir_source': 'ibn_kathir',  # for tafsir chunks
        
        # For Seerah:
        'event_name': 'Migration to Madinah',
        'chronological_order': 15,
        'year_hijri': 1,
    },
    
    # ===== CROSS-REFERENCES (for linking related content) =====
    'references': {
        'related_verses': ['2:183', '2:184'],      # Quranic references
        'related_hadiths': ['bukhari:8', 'muslim:16'],  # Hadith references
        'related_topics': ['fasting', 'ramadan'],
    },
    
    # ===== KNOWLEDGE GRAPH RELATIONSHIPS (Stage 10) =====
    'graph': {
        # Outgoing edges (forward relationships)
        'has_tafsir': ['tafsir_2:183_ibn_kathir', 'tafsir_2:183_tabari'],
        'related_to': ['2:182', '2:184'],
        'mentions_concept': ['concept_fasting', 'concept_ramadan'],
        'part_of_surah': ['surah_2'],
        'narrated_by': ['narrator_abu_huraira'],  # for hadith
        'supports_ruling': ['fiqh_hanafi_fasting_001'],  # for hadith
        'commentary_on': ['verse_2:183'],  # for tafsir (backward link)
        'authored_by': ['scholar_ibn_kathir'],  # for tafsir/fiqh
        'student_of': ['scholar_malik'],  # for scholars
        'cites_hadith': ['bukhari:1891'],  # for fiqh
    }
}
```

**Benefits of this approach:**

1. **No sparse fields**: Only populate `source_metadata` fields relevant to that source type
2. **Easy filtering**: Can filter by `source_type` + specific metadata fields
3. **Extensibility**: Add new source-specific fields without cluttering top level
4. **Clear structure**: Core fields vs. metadata vs. references
5. **Efficient storage**: Qdrant efficiently handles nested objects
6. **Better queries**: Can do things like `source_type == 'hadith' AND source_metadata.authenticity_grade == 'sahih'`

### 3.4 Universal Ingestion Manager (`backend/ingestion.py`)

LlamaIndex IngestionPipeline orchestrator for all text types (called from admin endpoints):

- Detect text type (Hadith/Tafsir/Fiqh/etc.)
- Load raw data as LlamaIndex Documents
- Apply appropriate NodeParser from `backend/chunking.py`
- Configure transformations: Document → Nodes → Embeddings
- Use LlamaIndex's Ollama embeddings integration
- Store in Qdrant via `llama-index-vector-stores-qdrant`
- Support batch processing with progress tracking
- Exposed through admin UI, not standalone scripts

Example workflow:
```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document

pipeline = IngestionPipeline(
    transformations=[
        HadithNodeParser(),  # Custom chunker
        OllamaEmbedding(),   # Ollama embeddings
    ],
    vector_store=qdrant_store
)

documents = load_hadith_json()  # Returns List[Document]
pipeline.run(documents=documents)
```

### 3.5 Knowledge Graph Integration (Stage 10)

**Goal:** Enhance vector RAG with explicit relationship tracking for connected knowledge traversal.

#### Graph Schema Design

**Relationship Types:**

```python
# Quran relationships
verse → has_tafsir → tafsir
verse → related_to → verse (sequential, thematic)
verse → mentions_concept → concept
verse → part_of_surah → surah
verse → revealed_after → verse (chronological)

# Hadith relationships  
hadith → narrated_by → narrator
hadith → supports_ruling → fiqh_ruling
hadith → references_verse → verse
hadith → chain_includes → narrator (isnad chain)
hadith → from_collection → collection

# Tafsir relationships
tafsir → commentary_on → verse (backward link)
tafsir → authored_by → scholar
tafsir → cites_hadith → hadith
tafsir → interprets_concept → concept

# Fiqh relationships
fiqh_ruling → based_on_verse → verse
fiqh_ruling → based_on_hadith → hadith
fiqh_ruling → madhab_view → madhab
fiqh_ruling → authored_by → scholar
fiqh_ruling → related_ruling → fiqh_ruling (within/across madhahib)

# Scholar relationships
scholar → student_of → scholar
scholar → teacher_of → scholar
scholar → belongs_to_madhab → madhab
scholar → authored → text (tafsir/fiqh/book)
scholar → contemporary_of → scholar

# Concept relationships
concept → subconcept_of → concept (hierarchy)
concept → related_concept → concept
concept → mentioned_in_verse → verse
concept → discussed_in_hadith → hadith
```

#### Storage in Qdrant Payload

Relationships stored as arrays of node IDs in the `graph` field:

```python
# Example: Verse 2:255 (Ayat al-Kursi)
{
    "id": "verse_2:255",
    "vector": [...],
    "payload": {
        "text_content": "Allah! There is no deity except Him...",
        "source_type": "quran",
        "graph": {
            "has_tafsir": ["tafsir_2:255_ibn_kathir", "tafsir_2:255_tabari"],
            "related_to": ["2:254", "2:256"],
            "mentions_concept": ["concept_tawhid", "concept_divine_names"],
            "part_of_surah": ["surah_2"]
        }
    }
}

# Example: Tafsir by Ibn Kathir
{
    "id": "tafsir_2:255_ibn_kathir",
    "vector": [...],
    "payload": {
        "text_content": "This is the greatest verse in the Quran...",
        "source_type": "tafsir",
        "graph": {
            "commentary_on": ["verse_2:255"],  # backward link
            "authored_by": ["scholar_ibn_kathir"],
            "cites_hadith": ["bukhari:7376"]
        }
    }
}
```

#### Graph Manager (`backend/vectordb/graph_manager.py`)

Lightweight graph operations on top of Qdrant:

```python
class QdrantGraphManager:
    """Graph traversal and relationship management using Qdrant."""
    
    def get_neighbors(node_id: str, edge_type: str) -> List[Point]
    def get_all_tafsir_for_verse(verse_key: str) -> List[Point]
    def get_verse_for_tafsir(tafsir_id: str) -> Point
    def get_narrator_chain(hadith_id: str) -> List[Point]
    def get_madhab_rulings(topic: str) -> Dict[str, List[Point]]
    def find_path(start_id: str, end_id: str, max_depth: int) -> List[str]
    def get_scholar_lineage(scholar_id: str) -> List[Point]
```

#### Ingestion Modifications

Parsers extract relationships during chunking:

```python
class QuranNodeParser(NodeParser):
    def _chunk_verse(self, doc, verse_data):
        # Create verse node
        verse_node = TextNode(...)
        
        # Extract relationships
        relationships = {
            "has_tafsir": extract_tafsir_refs(verse_data),
            "related_to": get_sequential_verses(verse_data),
            "mentions_concept": extract_concepts(verse_data),
            "part_of_surah": [f"surah_{verse_data['surah_number']}"]
        }
        
        # Add to metadata
        verse_node.metadata["graph"] = relationships
        
        return verse_node
```

#### Enhanced Retrieval with Graph Traversal

Hybrid vector + graph search:

```python
def retrieve_with_graph_expansion(query: str, depth: int = 1):
    # 1. Vector search
    similar_nodes = vector_search(query, limit=5)
    
    # 2. Graph expansion
    expanded_nodes = []
    for node in similar_nodes:
        # Get all connected tafsir
        tafsir = graph.get_neighbors(node.id, "has_tafsir")
        expanded_nodes.extend(tafsir)
        
        # Get related verses
        related = graph.get_neighbors(node.id, "related_to")
        expanded_nodes.extend(related)
    
    # 3. Return enriched context
    return {
        "primary_results": similar_nodes,
        "graph_expansion": expanded_nodes
    }
```

## Phase 4: Multi-Backend Model Integration (LlamaIndex) ✅ COMPLETED

**See `RAG-plan.md` for detailed LlamaIndex integration patterns.**

### 4.1 Multi-Backend Architecture ✅

**Embedding Backends Supported:**

- ✅ **HuggingFace/SentenceTransformers** - Fast local embeddings with true batching (RECOMMENDED)
  - Direct inference, no network overhead
  - GPU acceleration when available
  - Models: `google/embeddinggemma-300m`, `BAAI/bge-base-en-v1.5`, etc.
  
- ✅ **Ollama** - Flexible API-based embeddings
  - Easy model management via `ollama pull`
  - Can run on remote servers
  - Models: `embeddinggemma:latest`, `nomic-embed-text:latest`
  
- ✅ **LM Studio** - OpenAI-compatible local server
  - GUI-based model management
  - OpenAI API compatibility
  - Models: `text-embedding-embeddinggemma-300m-qat`, `text-embedding-nomic-embed-text-v1.5`

**LLM Backends Supported:**

- ✅ **Ollama** - Default, flexible API-based LLMs
  - Models: `qwen2.5:3b`, `qwen3-vl-8b`
  
- ✅ **LM Studio** - OpenAI-compatible local LLMs
  - Models: `islamspecialist-pro-12b`, `qwen/qwen3-vl-8b`

### 4.2 LlamaIndex Multi-Backend Configuration (`backend/llama/llama_config.py`) ✅

**Dynamic Backend Selection:**

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.lmstudio import LMStudio
from llama_index.core import Settings

# Configure based on environment variables
def configure_llama_index(
    embedding_backend="huggingface",  # or "ollama" or "lmstudio"
    llm_backend="ollama",              # or "lmstudio"
    embedding_model=None,
    llm_model=None
):
    # HuggingFace (fast local)
    if embedding_backend == "huggingface":
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model or "google/embeddinggemma-300m",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    # Ollama (API-based)
    elif embedding_backend == "ollama":
        embed_model = OllamaEmbedding(
            model_name=embedding_model or "embeddinggemma",
            base_url="http://localhost:11434",
        )
    
    # LM Studio (OpenAI-compatible)
    elif embedding_backend == "lmstudio":
        embed_model = OpenAIEmbedding(
            model_name=embedding_model or "text-embedding-embeddinggemma-300m-qat",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            mode=OpenAIEmbeddingMode.SIMILARITY_MODE,
        )
    
    # Configure LLM similarly based on llm_backend
    Settings.embed_model = embed_model
    Settings.llm = llm
```

**Backend services with multi-backend support:** ✅

- ✅ `backend/llama/embeddings_service.py` - Dynamic backend loading (HuggingFace/Ollama/LM Studio)
- ✅ `backend/llama/llm_service.py` - Multi-backend LLM support with proper timeout handling
- ✅ `backend/llama/reranker_service.py` - Backend-aware reranking (Ollama/LM Studio)
- ✅ `backend/llama/llama_config.py` - Centralized multi-backend configuration
- ✅ All following LlamaIndex best practices

**Configuration System:** ✅

- ✅ `backend/core/config.py` - Environment-based configuration
- ✅ `.env` file for backend selection
- ✅ Separate embedding and LLM backend choices
- ✅ Model-specific settings per backend

**Helper functions available:**

- `configure_llama_index(embedding_backend, llm_backend, ...)` - Set global Settings with backend choice
- `get_embed_model(embedding_backend, model_name, ...)` - Get configured embedding model
- `get_llm(llm_backend, model, ...)` - Get configured LLM
- `check_ollama_connection()` - Verify Ollama is reachable (when using Ollama)
- `check_model_available(model_name)` - Check if Ollama model is loaded

**Performance Comparison:**

| Backend | Speed | Use Case |
|---------|-------|----------|
| HuggingFace | ⚡⚡⚡ Fastest | Production ingestion, local inference |
| Ollama | ⚡⚡ Fast | Development, flexible model management |
| LM Studio | ⚡⚡ Fast | GUI-based management, OpenAI compatibility |

**Admin API endpoints** (to be integrated into LangGraph Server):

- `GET /api/admin/models/status` - Check which models are loaded
- `POST /api/admin/models/pull/:model` - Pull a model from Ollama library

## Phase 5: RAG Orchestration with LangGraph ✅ COMPLETED

**See `RAG-plan.md` for detailed LangGraph state machine design and workflow examples.**

**See `STAGE5_COMPLETE.md` for comprehensive completion report.**

### 5.1 System Prompt Engineering (`backend/rag/prompts.py`) ✅

**IMPLEMENTED** - Comprehensive system prompt with 467 lines including:

**Identity & Methodology:**

- ✅ "You are an Islamic knowledge assistant grounded in authentic Sunni sources"
- ✅ "Always prioritize Quran and authentic Hadith (Sahih al-Bukhari, Sahih Muslim)"
- ✅ "Present balanced views from the four madhahib when relevant"
- ✅ "Cite specific sources with book, chapter, and number references"

**Response Framework:**

- ✅ Connect general questions to Islamic perspective when possible
- ✅ For non-Islamic questions: briefly answer then relate to Islamic teachings if relevant
- ✅ Structure: Direct answer → Quranic/Hadith evidence → Scholarly opinions → Practical application

**Guardrails:**

- ✅ Never contradict clear Quranic/Hadith teachings
- ✅ Acknowledge scholarly differences respectfully
- ✅ Avoid sectarian polemics
- ✅ Clearly distinguish authentic from weak narrations

### 5.2 LangGraph Workflow (`backend/rag/rag_graph.py`) ✅

**IMPLEMENTED** - Full StateGraph workflow (300 lines) with conditional routing:

```python
from langgraph.graph import StateGraph

workflow = StateGraph(state_schema=RAGState)

# Add nodes (7 workflow nodes)
workflow.add_node("classify_query", classify_query_node)
workflow.add_node("expand_query", expand_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rank_context", rank_context_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("format_citations", format_citations_node)

# Add conditional routing by question type
workflow.add_conditional_edges(
    "classify_query",
    route_by_question_type,
    {
        "fiqh": "expand_query",      # Retrieve from all 4 madhahib
        "aqidah": "expand_query",     # Prioritize Quran + Sahih Hadith
        "tafsir": "expand_query",     # Verses + commentary
        "hadith": "expand_query",     # Hadith + related verses
        "general": "expand_query"     # Broad search
    }
)
```

**Workflow nodes implemented:**
- ✅ `classify_query`: Determine question type (fiqh/aqidah/tafsir/hadith/general)
- ✅ `expand_query`: Reformulate to be Islam-centric
- ✅ `retrieve`: Call LlamaIndex query engine with intelligent filtering
- ✅ `rank_context`: Apply authenticity weighting (Quran 1.0 → Seerah 0.60)
- ✅ `generate_response`: Use Ollama LLM with context
- ✅ `format_citations`: Structure source references

**Implementation notes:**
- Graph compiled without custom checkpointer (LangGraph Server provides automatic persistence)
- State management with thread-based conversations
- Async execution with streaming support

### 5.3 RAG Context Framing (`backend/rag/context_formatter.py`) ✅

**IMPLEMENTED** - Intelligent context formatter (502 lines) with authenticity ranking:

1. ✅ **Query expansion:** Reformulate user queries to be Islam-centric

   - "What is charity?" → "What is charity (Sadaqah/Zakat) in Islam?"

2. ✅ **Context ranking:** Prioritize sources by authenticity

   - Quran (1.0) > Sahih Hadith (0.85) > Aqidah (0.75) > Tafsir (0.70) > Usul (0.70) > Fiqh (0.65) > Seerah (0.60)

3. ✅ **Multi-perspective retrieval:** Fetch opinions from multiple madhahib for fiqh questions
4. ✅ **Cross-referencing:** Include related Quranic verses for Hadith, and vice versa
5. ✅ **Context templates:** Implemented for all query types
```python
context_template = """
Based on authentic Islamic sources:

PRIMARY EVIDENCE:
{quran_verses}
{sahih_hadiths}

SCHOLARLY INTERPRETATION:
{tafsir_excerpts}

JURISPRUDENTIAL VIEWS:
{fiqh_rulings_by_madhab}

PRACTICAL GUIDANCE:
{ethical_hadith}
"""
```


### 5.4 Smart Retrieval Strategy (`backend/rag/retrieval.py`) ✅

**IMPLEMENTED** - LlamaIndex query engine wrapper (566 lines) with intelligent filtering:

1. ✅ **Query classification:** Identify question type (fiqh/aqidah/tafsir/hadith/general) - handled by LangGraph node
2. ✅ **Multi-query expansion:** Generate 2-3 related queries - handled by LangGraph node
3. ✅ **Filtered search:** Use LlamaIndex metadata filters for source types based on question
4. ✅ **Hybrid ranking:** Combine semantic similarity + authenticity weighting
5. ✅ **Madhab-aware responses:** For fiqh questions, retrieve from all 4 madhahib

Implemented LlamaIndex integration:
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Create retriever with filters
retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=10,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(key="source_type", value="hadith"),
            MetadataFilter(key="source_metadata.authenticity_grade", value="sahih")
        ]
    )
)

query_engine = RetrieverQueryEngine(retriever=retriever)
```

### 5.5 LangGraph Server & Admin APIs ✅

**IMPLEMENTED** - Complete server setup and management endpoints:

- ✅ `backend/api/server.py` (195 lines) - LangGraph Server entry point
- ✅ `backend/api/admin/ingestion_api.py` (177 lines) - Data ingestion endpoints
- ✅ `backend/api/admin/collection_api.py` (284 lines) - Collection management
- ✅ `backend/api/admin/models_api.py` (228 lines) - Model status and health checks
- ✅ `langgraph.json` - LangGraph Server configuration

**Date Completed:** October 28, 2025  
**Total Lines:** ~2,900+ lines across 12 files  
**Test Coverage:** 7/7 tests passing

### Key Architecture Features

1. **Multi-Backend Flexibility:**
   - 3 embedding backends: HuggingFace (fast local), Ollama (API), LM Studio (OpenAI-compatible)
   - 2 LLM backends: Ollama, LM Studio
   - Dynamic selection via environment variables
   - Proper timeout handling for all backends

2. **Intelligent RAG Workflow:**
   - Query complexity analysis (simple vs. complex)
   - Query classification (fiqh, aqidah, tafsir, hadith, general)
   - Query expansion for better retrieval
   - Source-specific retrieval strategies
   - Authenticity-based ranking (Quran 1.0 → Seerah 0.60)
   - Madhab-aware fiqh responses
   - Streaming generation with cancellation support

3. **Production-Ready Ingestion:**
   - Streaming batch processing (prevents memory exhaustion)
   - Progress tracking with callbacks
   - Auto-detection of embedding dimensions
   - Custom NodeParsers for each Islamic text type
   - Background task management with single-job queue

4. **Comprehensive Admin API:**
   - Collection management (list, stats, delete, clear, export, search)
   - Ingestion management (upload, track, cancel)
   - Model status and health checks (Qdrant, Ollama, LM Studio)
   - Task tracking for long-running operations

5. **LangGraph Server Integration:**
   - Automatic API endpoints from graph definition
   - Built-in streaming via Server-Sent Events
   - Thread-based conversation management
   - State persistence with checkpointing
   - No Flask/FastAPI needed for chat endpoints

### With Knowledge Graph Enhancement:

- **Hybrid retrieval:** Combines semantic vector search with explicit relationship traversal
- **Complete context:** Get ALL related content (e.g., all tafsir for a verse), not just top-k similar
- **Evidence chains:** Trace provenance from verse → hadith → fiqh ruling
- **Scholar networks:** Navigate teacher-student chains and scholarly lineages
- **Narrator verification:** Complete isnad chains for hadith authenticity
- **Concept exploration:** Discover thematic connections across texts
- **Madhab comparison:** Cross-reference rulings with their evidence across schools
- **Interactive navigation:** Browse knowledge graph via admin UI
- **Richer answers:** Relationship-aware responses with deeper context
