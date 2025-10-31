# Islamic Chatbot RAG Implementation Plan

**Note:** This project now uses **LlamaIndex** for indexing/retrieval and **LangGraph** for RAG orchestration. See `RAG-plan.md` for detailed implementation architecture, patterns, and code examples.

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
2. **Modular Architecture:** Python packages with `__init__.py` files for proper module organization
3. **Import Style:** `from backend.core.config import Config`, `from backend.vectordb.qdrant_manager import QdrantManager`
4. **Separation of Concerns:** Frontend (React), Backend (LangGraph Server), RAG (LlamaIndex + LangGraph), Data (JSON files), Infrastructure (Docker)
5. **LlamaIndex Integration:** All indexing, retrieval, and query operations use LlamaIndex
6. **LangGraph Orchestration:** Complex RAG workflow managed by LangGraph StateGraph with native API endpoints
7. **Ollama Integration:** All models (embedding, reranker, chat) hosted through Ollama, accessed via LlamaIndex
8. **Admin UI:** All ingestion, migration, and management functionality accessible through React admin panel (no standalone scripts)
9. **Existing Files Migration (Stage 4 Complete):**
   - `embeddings_manager.py` → `backend/llama/embeddings_service.py` (wrapper around LlamaIndex OllamaEmbedding)
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

## Phase 6: Frontend Development

### 6.1 React Application Setup

**Initialize Vite + React:**

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install react-router-dom @langchain/langgraph-sdk axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 6.2 Core Components

**frontend/src/App.jsx** - Main application with routing:

- `/` - Chat interface
- `/admin` - Admin dashboard

**frontend/src/pages/Chat.jsx** - Main chat interface:

- Message history display with conversation threads
- Chat input box
- Source citations for each response
- Real-time streaming via LangGraph SDK
- Thread management (new conversation, resume thread)

**frontend/src/pages/Admin.jsx** - Admin dashboard with tabs:

- **Ingestion Tab:** Upload JSON files, trigger ingestion, show progress
- **Collections Tab:** View Qdrant collections, clear/export data
- **Models Tab:** Check Ollama model status, pull new models

**frontend/src/api/client.js** - Dual API client:

- LangGraph SDK client for chat operations:
  ```javascript
  import { Client } from "@langchain/langgraph-sdk";
  
  const client = new Client({
    apiUrl: "http://localhost:8123"
  });
  
  // Stream chat responses
  const streamResponse = async (message, threadId) => {
    const stream = client.runs.stream(
      threadId,
      "rag_assistant", // graph name
      { input: { messages: [message] } }
    );
    
    for await (const event of stream) {
      // Handle streaming events
    }
  };
  ```
- Axios instance for admin operations:
  ```javascript
  const adminClient = axios.create({
    baseURL: "http://localhost:8123/api/admin"
  });
  ```

### 6.3 Admin UI Functionality

**IngestionPanel.jsx:**

- File upload for Quran/Hadith/Tafsir JSON
- Dropdown to select source type
- Progress bar for ingestion
- Success/error notifications

**CollectionManager.jsx:**

- List all Qdrant collections
- Show collection stats (point count, vector size)
- Clear/delete collection buttons
- Export collection to JSON

**ModelStatus.jsx:**

- Display Ollama models status (loaded/not loaded)
- Show model details (size, parameters)
- Pull model button
- Health check indicator

## Phase 7: Implementation Steps

**See `RAG-plan.md` for detailed implementation guide and code examples.**

### Step 1: Install dependencies

```bash
# LlamaIndex core and integrations
pip install llama-index-core
pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-ollama
pip install llama-index-llms-ollama

# LangGraph and LangChain
pip install langgraph
pip install langchain-core
pip install langchain-community
pip install langchain-ollama
```

### Step 2: Backend restructuring ✅ COMPLETE (Stage 4)

1. Create modular `backend/` structure with Python packages:
   ```bash
   backend/
   ├── core/          # Configuration, models, utilities ✅
   ├── llama/         # LlamaIndex multi-backend integration ✅
   ├── vectordb/      # Vector database operations ✅
   ├── ingestion/     # Data pipeline with streaming ✅
   ├── rag/           # RAG workflow (Stage 5)
   ├── api/           # Server & endpoints (Stage 5)
   └── tests/         # Test suite ✅
   ```

2. **Core modules** (`backend/core/`) ✅:
   - `config.py` - Multi-backend configuration management
   - `models.py` - Pydantic schemas (state, API, documents)
   - `utils.py` - Utilities & logging

3. **LlamaIndex multi-backend integration** (`backend/llama/`) ✅:
   - `llama_config.py` - Dynamic backend configuration (HuggingFace/Ollama/LM Studio)
   - `embeddings_service.py` - Multi-backend embedding wrapper (3 backends)
   - `llm_service.py` - Multi-backend LLM wrapper (2 backends) with timeout handling
   - `reranker_service.py` - Backend-aware LLM reranking

4. **Vector database** (`backend/vectordb/`) ✅:
   - `qdrant_manager.py` - Qdrant + LlamaIndex VectorStore integration with enhanced schema

5. **Ingestion pipeline** (`backend/ingestion/`) ✅:
   - `parsers.py` - LlamaIndex NodeParsers (Quran + architecture for others)
   - `chunking.py` - Text chunking utilities
   - `ingestion.py` - Streaming LlamaIndex IngestionPipeline (batch processing)
   - `migrate_data.py` - Data migration utility

6. **Ingestion & Search Scripts** ✅:
   - `ingest_quran.py` - Full Quran ingestion with auto vector size detection
   - `search_quran.py` - Interactive search with backend awareness

7. **RAG workflow** (`backend/rag/`) ✅ **COMPLETED Stage 5**:
   - ✅ `rag_graph.py` (300 lines) - LangGraph StateGraph definition
   - ✅ `rag_nodes.py` (410 lines) - Individual workflow nodes
   - ✅ `retrieval.py` (566 lines) - LlamaIndex query engine wrapper
   - ✅ `context_formatter.py` (502 lines) - Context formatting
   - ✅ `prompts.py` (467 lines) - System prompts and templates

8. **API & Server** (`backend/api/`) ✅ **COMPLETED Stage 5**:
   - ✅ `server.py` (195 lines) - LangGraph Server setup
   - ✅ `admin/ingestion_api.py` (177 lines) - Ingestion endpoints
   - ✅ `admin/collection_api.py` (284 lines) - Collection management
   - ✅ `admin/models_api.py` (228 lines) - Model status endpoints
   - ✅ `langgraph.json` - LangGraph Server configuration (root level)
   - ✅ `test_rag_workflow.py` (176 lines) - Comprehensive test script

9. **Tests** (`backend/tests/`) ✅:
   - `test_stage4.py` - Stage 4 ingestion tests
   - `test_imports.py` - Import verification

### Step 3: Update docker-compose.yml

- Enable Ollama service
- Configure model pulling on startup
- Add LangGraph Server service
- Configure networking between services (Ollama, Qdrant, LangGraph Server)
- Set up volumes for state persistence
- Set environment variables

### Step 4: Frontend development

1. Initialize Vite React app in `frontend/`
2. Set up routing and layout
3. Install LangGraph SDK (`@langchain/langgraph-sdk`)
4. Build chat interface components with streaming support
5. Build admin panel components
6. Connect to LangGraph Server via SDK for chat
7. Connect to admin API via axios for management operations

### Step 5: Data model migration & ingestion ✅ COMPLETE (Stage 4)

- ✅ Updated payload schema in `backend/vectordb/qdrant_manager.py` (new fields OPTIONAL and additive)
- ✅ Ensured compatibility with LlamaIndex Document metadata structure
- ✅ Added filtered search methods by source/madhab/authenticity
- ✅ Created migration utility (`backend/ingestion/migrate_data.py`) to enrich existing Quran entries:
  - Add `book_title`: "The Noble Quran"
  - Add `book_title_arabic`: "القرآن الكريم"
  - Add `author`: "Allah (Revealed)"
  - Add `topic_tags`: Extract from metadata or generate based on content
  - Set `source_type` to standardized value: "quran"
- ✅ Migration only updates metadata, NO re-embedding or re-chunking needed
- ✅ Created streaming ingestion pipeline (`backend/ingestion/ingestion.py`):
  - Batch-wise processing to prevent memory exhaustion
  - Progress tracking with visual feedback
  - Support for all embedding backends
  - Auto-detects embedding dimensions
- ✅ Created ingestion scripts:
  - `ingest_quran.py` - Full Quran ingestion with backend flexibility
  - `search_quran.py` - Search with backend awareness
- ✅ Multi-backend configuration system:
  - Environment-based backend selection
  - Separate embedding and LLM backends
  - Proper timeout handling for all backends

### Step 6: Ingest Tier 1 texts

Via admin UI using LlamaIndex IngestionPipeline:

- Process and upload Sahih al-Bukhari and Sahih Muslim
- Process and upload chosen Tafsir
- Process and upload Riyad al-Salihin and Seerah

### Step 7: Test and iterate

- Test LangGraph Server endpoints with diverse question types
- Test conversation threading and state persistence
- Test streaming responses
- Evaluate LangGraph workflow execution and state transitions
- Use LangGraph Studio for workflow debugging and visualization
- Evaluate answer quality and source grounding
- Adjust retrieval weights and filters in `backend/retrieval.py`
- Refine LangGraph routing logic in `backend/rag_graph.py`
- Refine prompts in `backend/prompts.py`
- Monitor LlamaIndex query performance

### Step 8: Ingest Tier 2 & 3 (gradual expansion)

Via admin UI:

- Add remaining Hadith collections
- Add Fiqh texts from all madhahib
- Add methodology and specialized texts

## Key Design Decisions

1. **LlamaIndex for RAG foundation:** Robust indexing, retrieval, and query engine with production-ready features
2. **Multi-backend flexibility:** Support for 3 embedding backends (HuggingFace/Ollama/LM Studio) and 2 LLM backends (Ollama/LM Studio)
3. **HuggingFace for performance:** Fast local embeddings with true batching, GPU acceleration when available
4. **Streaming ingestion:** Batch-wise processing prevents memory exhaustion, enables resumable operations
5. **LangGraph Server for orchestration:** Native API server with state machine workflow, built-in streaming, and persistence - no Flask needed
6. **Modular architecture:** Python packages with `__init__.py` files - clear separation of concerns (core, llama, vectordb, ingestion, rag, api, tests)
7. **Environment-based configuration:** Easy backend switching via `.env` file without code changes
8. **Admin UI over scripts:** All management functionality (ingestion, migration, collection management) accessible through interfaces
9. **LangGraph native endpoints:** Automatic API generation from graph definition with built-in streaming
10. **Metadata-rich chunks:** Every chunk carries full provenance for citation (LlamaIndex Documents/Nodes)
11. **Authenticity weighting:** Sahih sources rank higher than Daif in retrieval (handled by LangGraph context ranking)
12. **Madhab balance:** Fiqh queries fetch from all 4 schools automatically (LangGraph conditional routing)
13. **LLM as synthesizer:** LLM doesn't invent, only synthesizes retrieved sources
14. **Progressive ingestion:** Start with Tier 1, expand to Tier 2/3 based on coverage gaps
15. **Arabic + English:** Maintain both for authenticity and accessibility
16. **Workflow transparency:** LangGraph state machine makes RAG logic explicit and debuggable
17. **Built-in state management:** LangGraph checkpointing handles conversation threads natively
18. **LangGraph Studio:** Visual debugging and workflow inspection out of the box
19. **Proper timeout handling:** Backend-specific timeout configuration prevents premature failures
20. **Auto-dimension detection:** Embedding dimensions automatically detected for any model

## Backend Restructuring (Completed)

As of Stage 4, the backend has been restructured into a modular architecture with proper Python packages:

### Structure Transformation

**From:** Flat structure with mixed concerns
**To:** Modular packages with clear separation

### New Import Patterns

```python
# Core functionality
from backend.core.config import Config
from backend.core.models import SourceType, QdrantPayload, create_qdrant_payload
from backend.core.utils import setup_logging, ProgressTracker

# LlamaIndex/Ollama integration
from backend.llama.llama_config import configure_llama_index, get_embed_model, get_llm
from backend.llama.embeddings_service import EmbeddingsService
from backend.llama.llm_service import LLMService
from backend.llama.reranker_service import RerankerService

# Vector database
from backend.vectordb.qdrant_manager import QdrantManager

# Ingestion pipeline
from backend.ingestion.parsers import QuranNodeParser, HadithNodeParser, TafsirNodeParser
from backend.ingestion.chunking import parse_tafsir_html, chunk_plain_text
from backend.ingestion.ingestion import IngestionManager, ingest_quran
from backend.ingestion.migrate_data import DataMigration

# RAG components (Stage 5)
from backend.rag.rag_graph import create_rag_graph
from backend.rag.retrieval import create_retriever
from backend.rag.prompts import SYSTEM_PROMPT

# API/Server (Stage 5)
from backend.api.server import create_app
```

### Benefits Achieved

1. **Clear Separation:** Each package has a single, well-defined responsibility
2. **Easier Navigation:** Intuitive file organization makes code easy to find
3. **Better Imports:** Type hints and autocomplete work more effectively
4. **Scalable:** Easy to add new components without affecting existing code
5. **Testable:** Isolated packages are easier to test independently
6. **Stage 5 Ready:** Structure prepared for RAG and API components

### Verification

All imports verified and tested:
```bash
# Fast import verification (2 seconds)
python -m backend.tests.test_imports

# Results: 5/5 tests passed
✓ Core imports successful
✓ Llama imports successful
✓ VectorDB imports successful
✓ Ingestion imports successful
✓ Config paths correct
```

## Expected Outcomes

- **Production-ready RAG system:** LlamaIndex + LangGraph Server provide battle-tested infrastructure
- **Modern full-stack application:** React frontend + LangGraph Server + LlamaIndex + Ollama + Qdrant
- **Simplified architecture:** No Flask/FastAPI needed - LangGraph Server handles all API requirements
- **Native streaming:** Built-in streaming support via LangGraph Server with Server-Sent Events
- **User-friendly admin panel:** Upload and ingest texts via LlamaIndex pipelines, manage collections, monitor models
- **Comprehensive knowledge base:** Covering 15-20 foundational texts across Quran, Hadith, Tafsir, Fiqh, Seerah
- **Intelligent retrieval:** Balances authenticity, relevance, and madhab diversity with LlamaIndex metadata filtering
- **Sophisticated workflow:** LangGraph state machine orchestrates query classification, expansion, retrieval, ranking, and generation
- **Persistent conversations:** Thread-based conversation management with automatic checkpointing
- **Islam-first responses:** Ground general questions in Islamic teachings using RAG
- **Transparent citations:** Every claim backed by specific source references (Surah:Verse, Hadith number, etc.)
- **Scalable architecture:** Easy to add new texts, update models, extend workflow nodes
- **Self-hosted:** All models run locally through Ollama - no external API dependencies
- **Debuggable workflows:** LangGraph Studio visualization and state inspection for troubleshooting
- **Type-safe state management:** Pydantic schemas for all state transitions
