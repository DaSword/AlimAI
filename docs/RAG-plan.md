# RAG System Implementation Plan

## Overview

Build a production-ready RAG system using **LangGraph Server** (orchestration) + **LlamaIndex** (ingestion/retrieval) + **Ollama** (embeddings/LLM). Focus: Backend only, multi-source Islamic knowledge base with intelligent routing.

---

## Phase 1: Multi-Backend Integration & LlamaIndex Setup ✅ COMPLETED

**Goal:** Set up multiple embedding and LLM backends, configure LlamaIndex with flexible backend selection and Qdrant integration.

### Tasks

1. **Install Dependencies** (`requirements.txt`) ✅
   ```
   # LlamaIndex Core
   llama-index-core>=0.11.0
   llama-index-vector-stores-qdrant>=0.3.0
   
   # Embedding Backends
   llama-index-embeddings-ollama>=0.3.0
   llama-index-embeddings-huggingface>=0.3.0
   llama-index-embeddings-openai>=0.2.0  # For LM Studio
   sentence-transformers>=2.2.0
   torch>=2.0.0
   
   # LLM Backends
   llama-index-llms-ollama>=0.3.0
   llama-index-llms-lmstudio>=0.2.0
   
   # RAG Orchestration
   langgraph>=0.2.0
   langchain-core>=0.3.0
   langchain-ollama>=0.2.0
   ```

2. **Configure Multi-Backend System** ✅

   **Embedding Backends:**
   - ✅ **HuggingFace/SentenceTransformers** - Fast local embeddings with true batching (RECOMMENDED)
     - Models: `google/embeddinggemma-300m`, `BAAI/bge-base-en-v1.5`
     - Direct inference, GPU acceleration, no network overhead
   - ✅ **Ollama** - Flexible API-based embeddings
     - Models: `embeddinggemma:latest`, `nomic-embed-text:latest`
     - Easy model management, can run remotely
   - ✅ **LM Studio** - OpenAI-compatible local server
     - Models: `text-embedding-embeddinggemma-300m-qat`, `text-embedding-nomic-embed-text-v1.5`
     - GUI-based management, OpenAI API compatibility

   **LLM Backends:**
   - ✅ **Ollama** - Default, flexible API-based LLMs
     - Models: `qwen2.5:3b`, `qwen3-vl-8b`
   - ✅ **LM Studio** - OpenAI-compatible local LLMs
     - Models: `islamspecialist-pro-12b`, `qwen/qwen3-vl-8b`

3. **Create Multi-Backend LlamaIndex Configuration** (`backend/llama/llama_config.py`) ✅

   - ✅ Dynamic backend selection based on environment variables
   - ✅ Configure `HuggingFaceEmbedding`, `OllamaEmbedding`, and `OpenAIEmbedding` (for LM Studio)
   - ✅ Configure `Ollama` and `LMStudio` LLMs
   - ✅ Helper functions: `get_embed_model(backend, ...)`, `get_llm(backend, ...)`
   - ✅ Connection checking: `check_ollama_connection()`, `check_model_available()`
   - ✅ Proper timeout handling for all backends
   - ✅ Set global `Settings.embed_model` and `Settings.llm` based on configuration

4. **Create Multi-Backend Embeddings Service** (`backend/llama/embeddings_service.py`) ✅

   - ✅ Dynamic backend loading (HuggingFace/Ollama/LM Studio)
   - ✅ Implement efficient batch embedding generation
   - ✅ Auto-detect device for HuggingFace (CPU/GPU)
   - ✅ Tested with Islamic texts (Arabic + English)
   - ✅ Support for HuggingFace tokens (gated models)

5. **Create Multi-Backend LLM Service** (`backend/llama/llm_service.py`) ✅

   - ✅ Support Ollama and LM Studio backends
   - ✅ Support streaming responses
   - ✅ Handle context window management
   - ✅ Proper timeout configuration (request_timeout + timeout)
   - ✅ Graceful error handling for memory constraints

6. **Create Backend-Aware Reranker Service** (`backend/llama/reranker_service.py`) ✅
   
   - ✅ Uses LlamaIndex `LLMRerank` postprocessor
   - ✅ Support for Ollama and LM Studio backends
   - ✅ LLM-based reranking following official examples
   - ✅ Works with `NodeWithScore` objects
   - ✅ Graceful fallback when memory limited

7. **Create Configuration Management** (`backend/core/config.py`) ✅
   
   - ✅ Environment-based backend selection
   - ✅ Separate `EMBEDDING_BACKEND` and `LLM_BACKEND` configuration
   - ✅ Model-specific settings per backend
   - ✅ Timeout configuration per backend
   - ✅ `.env.example` with all configuration options

### Deliverables

- ✅ Updated `requirements.txt` with all backend dependencies
- ✅ Multi-backend configuration system
- ✅ `backend/llama/llama_config.py` with dynamic backend selection
- ✅ `backend/llama/embeddings_service.py` - Multi-backend embedding wrapper
- ✅ `backend/llama/llm_service.py` - Multi-backend LLM wrapper
- ✅ `backend/llama/reranker_service.py` - Backend-aware reranking
- ✅ `backend/core/config.py` - Centralized configuration
- ✅ `.env.example` - Complete configuration template

### Acceptance Criteria

- ✅ All three embedding backends work (HuggingFace/Ollama/LM Studio)
- ✅ Both LLM backends work (Ollama/LM Studio)
- ✅ Backend switching works via environment variables
- ✅ LlamaIndex successfully connects to all backends
- ✅ Embedding generation works for Arabic and English text
- ✅ Vector dimensions auto-detected for any model
- ✅ LLM generates chat completions via all backends
- ✅ Reranker works with both LLM backends
- ✅ All services handle memory constraints gracefully
- ✅ Timeout issues resolved for all backends
- ✅ HuggingFace backend provides fastest ingestion
- ✅ Configuration properly documented in `.env.example`

### Implementation Notes

- **HuggingFace (RECOMMENDED)**: Fastest for ingestion, true batching, GPU acceleration
- **Ollama**: Great for development, flexible model management
- **LM Studio**: GUI-based, OpenAI-compatible, good for local testing
- Using `embeddinggemma` across backends for consistency
- Timeout handling: `request_timeout` (HTTP) + `timeout` (overall operation)
- Reranker uses `LLMRerank` from `llama_index.core.postprocessor.llm_rerank`
- All implementations follow official LlamaIndex documentation
- Based on: https://developers.llamaindex.ai/python/examples/workflow/rag/
- Performance: HuggingFace > Ollama ≈ LM Studio for embeddings

---

## Phase 2: Custom NodeParsers for Multi-Source Ingestion

**Goal:** Create LlamaIndex NodeParsers for each Islamic text type with proper metadata extraction.

### Tasks

1. **Create Base Islamic NodeParser** (`backend/parsers/base_parser.py`)

   - Extend LlamaIndex `NodeParser` base class
   - Common methods for metadata extraction
   - Chunk size configuration (500-2000 chars)
   - Arabic/English text handling

2. **Create QuranNodeParser** (`backend/parsers/quran_parser.py`)

   - Migrate logic from `quran_chunker.py`
   - Parse verse + tafsir data
   - Extract: verse_key, surah_number, verse_number, chapter_name, arabic_text, english_text
   - Handle tafsir chunks separately with `chunk_type="tafsir"`
   - Preserve semantic boundaries

3. **Create HadithNodeParser** (`backend/parsers/hadith_parser.py`)

   - Parse individual hadith with isnad (chain) + matn (text)
   - Extract: hadith_number, book_name, chapter_name, narrator_chain, authenticity_grade
   - Metadata: `source_metadata.hadith_number`, `source_metadata.authenticity_grade`
   - Support Bukhari/Muslim format from sunnah.com API

4. **Create TafsirNodeParser** (`backend/parsers/tafsir_parser.py`)

   - Parse verse-by-verse commentary
   - Extract: verse_key, tafsir_author, commentary_text
   - Link to original verse via `references.related_verses`
   - Handle HTML content if needed

5. **Create FiqhNodeParser** (`backend/parsers/fiqh_parser.py`)

   - Parse fiqh rulings by topic
   - Extract: madhab, ruling_category, ruling_text
   - Metadata: `source_metadata.madhab`, `source_metadata.ruling_category`
   - Hierarchical structure: Book → Chapter → Section → Ruling

6. **Create SeerahNodeParser** (`backend/parsers/seerah_parser.py`)

   - Parse chronological events
   - Extract: event_name, year_hijri, location, participants
   - Metadata: `source_metadata.chronological_order`, `source_metadata.year_hijri`

### Deliverables

- Base parser: `backend/parsers/base_parser.py`
- Specialized parsers: `quran_parser.py`, `hadith_parser.py`, `tafsir_parser.py`, `fiqh_parser.py`, `seerah_parser.py`
- Unit tests for each parser with sample data

### Acceptance Criteria

- Each parser extends LlamaIndex `NodeParser`
- Parsers extract all relevant metadata for each source type
- Generated Nodes conform to universal schema
- Chunk sizes appropriate for embeddings (500-2000 chars)
- Arabic + English text preserved correctly

---

## Phase 3: Universal Ingestion Pipeline ✅ COMPLETED

**Goal:** Build LlamaIndex IngestionPipeline to process all Islamic text types with streaming support.

### Tasks

1. **Create Ingestion Manager** (`backend/ingestion/ingestion.py`) ✅

   - ✅ Detect source type from JSON structure
   - ✅ Load JSON as LlamaIndex `Document` objects
   - ✅ Apply appropriate NodeParser based on source type
   - ✅ Configure transformations: Documents → Nodes → Embeddings → VectorStore
   - ✅ **Streaming batch processing** to prevent memory exhaustion
   - ✅ Progress tracking with visual feedback
   - ✅ Error handling and validation
   - ✅ Support for all embedding backends

2. **Create Document Loaders** (`backend/ingestion/ingestion.py`) ✅

   - ✅ `json_to_documents()`: Universal JSON to Document converter
   - ✅ Support for Quran format (verses + tafsir)
   - ✅ Architecture ready for Hadith, Tafsir, Fiqh, Seerah
   - ✅ Returns `List[Document]` with rich metadata
   - ✅ Handles nested source_metadata structure

3. **Build Ingestion Scripts** ✅

   - ✅ `ingest_quran.py`: Full Quran ingestion with auto-detection
   - ✅ `ingest_quran_sample.py`: Sample ingestion for testing
   - ✅ Arguments: file path, collection name, batch size
   - ✅ Progress bar for batch operations
   - ✅ Summary statistics (points ingested, time elapsed, embeddings/sec)
   - ✅ Backend-aware (works with all embedding backends)

4. **Create Search Functionality** ✅

   - ✅ `search_quran.py`: Interactive search script
   - ✅ Query engine integration with LlamaIndex
   - ✅ Backend-aware search
   - ✅ Source citations and scoring
   - ✅ Interactive and single-query modes

5. **Implement Streaming Ingestion** ✅

   - ✅ Load JSON once to get item count
   - ✅ Process in configurable batches (default 100 items)
   - ✅ Run pipeline for each batch separately
   - ✅ Memory-efficient, resumable operations
   - ✅ Clear batch from memory after processing

### Deliverables

- ✅ `backend/ingestion/ingestion.py` with streaming IngestionPipeline
- ✅ Document conversion utilities
- ✅ `ingest_quran.py` - Production ingestion script
- ✅ `search_quran.py` - Search functionality
- ✅ Successfully tested with Quran data (streaming + all backends)

### Acceptance Criteria

- [x] IngestionPipeline processes Quran data without errors
- [x] Embeddings generated via all backends (HuggingFace/Ollama/LM Studio)
- [x] Streaming prevents memory exhaustion on large datasets
- [x] Qdrant collection creation with auto-detected vector dimensions
- [x] Metadata fields populated correctly per universal schema
- [x] Can filter by `source_type="quran"` in Qdrant
- [x] Search functionality works with all backends
- [x] Progress tracking provides accurate feedback

---

## Phase 4: LangGraph Workflow Design ✅ COMPLETED

**Goal:** Build intelligent RAG workflow with query classification, retrieval, ranking, and generation.

**See `STAGE5_COMPLETE.md` for comprehensive completion report.**

### Tasks

1. ✅ **Define State Schema** (`backend/core/models.py`)

   - ✅ Create Pydantic models for RAG state
   - ✅ `RAGState`: messages, query_type, retrieval_results, context, response, metadata
   - ✅ `QueryType`: Enum (fiqh, aqidah, tafsir, hadith, general)
   - ✅ `RetrievalResult`: text, score, source_type, metadata

2. ✅ **Create System Prompts** (`backend/rag/prompts.py` - 467 lines)

   - ✅ Identity prompt: "Islamic knowledge assistant grounded in authentic sources"
   - ✅ Query classification prompt: Identify question type
   - ✅ Query expansion prompt: Reformulate to be Islam-centric
   - ✅ Generation prompt: Synthesize from sources with citations
   - ✅ Templates for each query type (fiqh, aqidah, tafsir, hadith, general)

3. ✅ **Implement Workflow Nodes** (`backend/rag/rag_nodes.py` - 410 lines)

   - ✅ `classify_query_node`: Use LLM to classify question type (fiqh/aqidah/tafsir/hadith/general)
   - ✅ `expand_query_node`: Reformulate query for better retrieval
   - ✅ `retrieve_node`: Call LlamaIndex query engine with filters
   - ✅ `rank_context_node`: Rerank by authenticity (Quran 1.0 → Seerah 0.60)
   - ✅ `generate_response_node`: LLM generates answer from context
   - ✅ `format_citations_node`: Structure source references

4. ✅ **Build RAG Graph** (`backend/rag/rag_graph.py` - 300 lines)

   - ✅ Create `StateGraph` with RAGState schema
   - ✅ Add nodes from `rag_nodes.py`
   - ✅ Conditional edges based on query type:
     - fiqh → retrieve from multiple madhahib
     - aqidah → prioritize Quran + Sahih Hadith
     - tafsir → retrieve verse + commentary
     - hadith → hadith + related verses
     - general → broad search
   - ✅ Compile graph with LangGraph Server-compatible checkpointing

5. ✅ **Create Context Formatter** (`backend/rag/context_formatter.py` - 502 lines)

   - ✅ Rank sources by authenticity (Quran 1.0 > Sahih Hadith 0.85 > ... > Seerah 0.60)
   - ✅ Structure context for LLM prompt
   - ✅ Templates for different query types
   - ✅ Include cross-references (related verses, hadiths)

### Deliverables

- ✅ `backend/core/models.py` with Pydantic state schemas
- ✅ `backend/rag/prompts.py` (467 lines) with all system prompts
- ✅ `backend/rag/rag_nodes.py` (410 lines) with workflow node implementations
- ✅ `backend/rag/rag_graph.py` (300 lines) with compiled StateGraph
- ✅ `backend/rag/context_formatter.py` (502 lines) with context structuring logic
- ✅ `backend/rag/retrieval.py` (566 lines) with LlamaIndex query engine wrapper

### Acceptance Criteria

- ✅ RAG workflow executes from query → response
- ✅ Query classification correctly identifies question types (5 types)
- ✅ Conditional routing works based on query type
- ✅ Retrieval node successfully queries LlamaIndex
- ✅ Generated responses include source citations
- ✅ State persists across conversation turns with thread management

**Date Completed:** October 28, 2025  
**Total Lines:** ~2,245 lines across RAG components

---

## Phase 5: Multi-Source Retrieval Strategy

**Goal:** Implement intelligent retrieval that adapts to query type and source characteristics.

### Tasks

1. **Create LlamaIndex Query Engine Wrapper** (`backend/retrieval.py`)

   - Wrap `VectorStoreIndex.as_query_engine()`
   - Support metadata filters via LlamaIndex `MetadataFilters`
   - Implement filtered search methods:
     - `retrieve_by_source_type(query, source_types: List[str])`
     - `retrieve_by_madhab(query, madhab: str)` for fiqh questions
     - `retrieve_by_authenticity(query, min_grade: str)` for hadith
     - `retrieve_quran_with_tafsir(query)` - verse + commentary
   - Multi-query expansion: generate 2-3 related queries

2. **Implement Hybrid Ranking** (`backend/ranking.py`)

   - Combine semantic similarity with authenticity weighting
   - Scoring formula: `final_score = similarity * authenticity_weight`
   - Authenticity weights:
     - Quran: 1.0
     - Sahih Hadith: 0.9
     - Hasan Hadith: 0.7
     - Tafsir (scholar): 0.6
     - Fiqh ruling: 0.5
   - Rerank results before passing to LLM

3. **Build Madhab-Aware Fiqh Retriever** (`backend/retrievers/fiqh_retriever.py`)

   - For fiqh questions, retrieve from all 4 madhahib
   - Parallel queries with madhab filters
   - Combine results with balanced representation
   - Format: "Hanafi: ..., Maliki: ..., Shafi'i: ..., Hanbali: ..."

4. **Create Cross-Reference Resolver** (`backend/retrievers/cross_reference.py`)

   - When retrieving hadith, also fetch related Quranic verses
   - When retrieving verse, fetch related hadiths
   - Use `references` metadata field for linking
   - Enrich context with cross-referenced sources

### Deliverables

- `backend/retrieval.py` with LlamaIndex query engine wrapper
- `backend/ranking.py` with authenticity-based reranking
- `backend/retrievers/fiqh_retriever.py` for madhab-aware queries
- `backend/retrievers/cross_reference.py` for source linking

### Acceptance Criteria

- Retrieval filters work correctly by source_type, madhab, authenticity
- Fiqh questions return perspectives from multiple madhahib
- Authenticity ranking prioritizes Quran > Sahih Hadith > other sources
- Cross-references enrich context with related sources
- Multi-query expansion improves retrieval recall

---

## Phase 6: LangGraph Server Setup ✅ COMPLETED

**Goal:** Deploy LangGraph Server with RAG workflow, admin endpoints, and streaming support.

**See `STAGE5_COMPLETE.md` for comprehensive completion report.**

### Tasks

1. ✅ **Create LangGraph Server Configuration** (`langgraph.json`)
   ```json
   {
     "dependencies": ["backend"],
     "graphs": {
       "rag_assistant": "backend.rag.rag_graph:graph"
     },
     "env": ".env"
   }
   ```

2. ✅ **Create Server Entry Point** (`backend/api/server.py` - 195 lines)

   - ✅ Initialize LangGraph Server
   - ✅ Register RAG workflow graph
   - ✅ LangGraph Server provides automatic persistence (no custom checkpointer needed)
   - ✅ Entry point function: `get_graph()`

3. ✅ **Build Admin Endpoints** (`backend/api/admin/`)

   - ✅ `ingestion_api.py` (177 lines): 
     - Ingestion file handlers
     - Progress tracking
     - Batch processing support
   - ✅ `collection_api.py` (284 lines):
     - List Qdrant collections
     - Collection statistics
     - Clear/delete collection operations
     - Export functionality
   - ✅ `models_api.py` (228 lines):
     - Check Ollama/LM Studio models
     - Health check for all services (Qdrant, Ollama, embeddings, LLM)
     - Model availability verification

4. ✅ **Update Docker Compose** (`docker-compose.yml`)

   - ✅ Added LangGraph Server service documentation/notes
   - ✅ Environment variables: `QDRANT_URL`, `OLLAMA_URL`, backend configuration
   - ✅ Networking: documented connections to Ollama and Qdrant
   - ✅ Notes on volumes for state persistence

5. ✅ **Environment Config** (`.env.example` already exists)
   ```
   QDRANT_URL=http://localhost:6333
   OLLAMA_URL=http://localhost:11434
   EMBEDDING_BACKEND=huggingface  # or ollama, lmstudio
   EMBEDDING_MODEL=google/embeddinggemma-300m
   LLM_BACKEND=ollama  # or lmstudio
   LLM_MODEL=qwen2.5:3b
   COLLECTION_NAME=islamic_knowledge
   ```


### Deliverables

- ✅ `langgraph.json` configuration file
- ✅ `backend/api/server.py` (195 lines) with LangGraph Server setup
- ✅ Admin API endpoints in `backend/api/admin/` (689 lines total)
  - ✅ `ingestion_api.py` (177 lines)
  - ✅ `collection_api.py` (284 lines)
  - ✅ `models_api.py` (228 lines)
- ✅ Updated `docker-compose.yml` with LangGraph Server notes
- ✅ `.env.example` with multi-backend configuration

### Acceptance Criteria

- ✅ LangGraph Server starts successfully with `langgraph dev --port 8123`
- ✅ LangGraph Server accessible at `http://localhost:8123`
- ⏸️ Main chat endpoint: `POST /runs/stream` returns streaming responses (requires ingested data)
- ✅ Admin endpoints implemented and functional
- ✅ Health check returns status of all connected services
- ✅ State management with thread-based conversations implemented
- ✅ Multi-backend support (HuggingFace/Ollama/LM Studio)

**Date Completed:** October 28, 2025  
**Total Lines:** ~884 lines across server and admin endpoints  
**Test Script:** `test_rag_workflow.py` (176 lines) with 7/7 tests passing

---

## Phase 7: Testing & Refinement

**Goal:** Test end-to-end RAG pipeline with diverse Islamic queries and refine for quality.

### Tasks

1. **Create Test Suite** (`tests/`)

   - `test_schema_migration.py`: Verify Quran data migration
   - `test_ingestion.py`: Test ingestion pipeline for each source type
   - `test_retrieval.py`: Test retrieval with various filters
   - `test_rag_workflow.py`: Test full RAG pipeline
   - `test_query_types.py`: Test classification and routing

2. **Build Test Query Dataset** (`tests/test_queries.json`)

   - Aqidah: "What are the pillars of Iman?"
   - Fiqh: "How do I pray Witr?" (should retrieve from all madhahib)
   - Tafsir: "What does Ayat al-Kursi mean?"
   - Hadith: "What did the Prophet say about intentions?"
   - General: "What is the purpose of life in Islam?"

3. **Create Evaluation Script** (`backend/evaluation/evaluate.py`)

   - Run test queries through RAG pipeline
   - Measure: retrieval accuracy, citation correctness, response quality
   - Log results with scores
   - Compare responses across iterations

4. **Refine System Based on Results**

   - Adjust retrieval weights in `backend/ranking.py`
   - Improve prompts in `backend/prompts.py`
   - Fine-tune context formatting in `backend/context_formatter.py`
   - Optimize chunk sizes if needed

5. **Create Testing Documentation** (`docs/TESTING.md`)

   - How to run tests
   - How to add new test cases
   - Expected results and benchmarks

### Deliverables

- Comprehensive test suite in `tests/`
- Test query dataset with expected behaviors
- Evaluation script for quality assessment
- Testing documentation

### Acceptance Criteria

- 90%+ of test queries return relevant responses
- All responses include verifiable source citations
- Fiqh questions return perspectives from multiple madhahib
- Retrieval correctly prioritizes Quran > Sahih Hadith
- No hallucinations (claims without source backing)
- Response time acceptable (<10s per query)

---

## Phase 8: Documentation & Finalization

**Goal:** Complete documentation and ensure system is production-ready.

### Tasks

1. **Update Build Plan** (`build-plan.md`)

   - Mark Stages 2-5 as complete
   - Update schema section with new design
   - Document LlamaIndex + LangGraph integration

2. **Update Pipeline Documentation** (`islam-pipeline.md`)

   - Reflect LangGraph Server architecture
   - Update ingestion pipeline section
   - Add RAG workflow diagram (text-based)

3. **Create Developer Guide** (`docs/DEVELOPER.md`)

   - System architecture overview
   - How to add new source types
   - How to modify RAG workflow
   - How to extend NodeParsers

4. **Create Operations Guide** (`docs/OPERATIONS.md`)

   - How to start/stop services
   - How to ingest new data
   - How to monitor performance
   - Troubleshooting common issues

5. **Create API Documentation** (`docs/API.md`)

   - Main chat endpoint usage
   - Admin endpoints reference
   - Request/response examples
   - Error codes and handling

### Deliverables

- Updated `build-plan.md` and `islam-pipeline.md`
- Developer guide: `docs/DEVELOPER.md`
- Operations guide: `docs/OPERATIONS.md`
- API documentation: `docs/API.md`

### Acceptance Criteria

- All documentation reflects current implementation
- Clear instructions for adding new sources
- Operations guide enables non-developers to manage system
- API documentation includes complete examples

---

## Implementation Order

1. ✅ **Phase 1** - Multi-Backend + LlamaIndex Setup (required for ingestion) - **COMPLETED**
2. ⏸️ **Phase 2** - NodeParsers (required for multi-source ingestion) - **Quran parser complete, others pending**
3. ✅ **Phase 3** - Ingestion Pipeline (get data into system) - **COMPLETED**
4. ✅ **Phase 4** - LangGraph Workflow (RAG logic) - **COMPLETED**
5. ⏸️ **Phase 5** - Multi-Source Retrieval (intelligent querying) - **Partially completed in Phase 4**
6. ✅ **Phase 6** - LangGraph Server (deployment) - **COMPLETED**
7. ⏸️ **Phase 7** - Testing & Refinement (quality assurance) - **Pending full data ingestion**
8. ⏸️ **Phase 8** - Documentation (finalization) - **In progress**

**Current Status:** Stage 5 complete (October 28, 2025). Backend RAG system fully operational. Ready for Stage 6 (Frontend Development).

---

## Key Design Decisions

1. **LlamaIndex for RAG Foundation**: Production-ready indexing, retrieval, and query engine
2. **LangGraph Server for Orchestration**: Native API with streaming, state management, no Flask needed
3. **Universal Schema**: Flexible metadata structure supports all source types
4. **Ollama Integration**: Self-hosted embeddings and LLM, no external API dependencies
5. **Source-Aware Routing**: Query classification drives retrieval strategy
6. **Authenticity Ranking**: Prioritize Quran > Sahih Hadith in context ranking
7. **Madhab Balance**: Fiqh queries automatically fetch from all 4 schools
8. **Cross-Referencing**: Enrich context with related verses/hadiths
9. **State Persistence**: Conversation threads maintained via LangGraph checkpointing
10. **Extensible Architecture**: Easy to add new sources, parsers, and retrieval strategies

---

## Success Metrics

### Completed ✅
- ✅ **Phase 1 Complete**: Multi-backend integration with LlamaIndex
  - ✅ HuggingFace embeddings (fast local, true batching)
  - ✅ Ollama embeddings (API-based, flexible)
  - ✅ LM Studio embeddings (OpenAI-compatible)
  - ✅ Ollama LLM service operational
  - ✅ LM Studio LLM service operational
  - ✅ Backend switching via environment variables
  - ✅ Proper timeout handling for all backends
- ✅ **Phase 3 Complete**: Universal ingestion pipeline
  - ✅ Streaming batch processing prevents memory issues
  - ✅ Works with all embedding backends
  - ✅ Auto-detects embedding dimensions
  - ✅ Progress tracking and statistics
  - ✅ Quran ingestion and search scripts working
- ✅ **Phase 4 Complete**: LangGraph workflow for RAG orchestration
  - ✅ RAGState schema with Pydantic models
  - ✅ System prompts and templates (467 lines)
  - ✅ 7 workflow nodes implemented (410 lines)
  - ✅ StateGraph with conditional routing (300 lines)
  - ✅ Context formatter with authenticity ranking (502 lines)
  - ✅ LlamaIndex query engine wrapper (566 lines)
  - ✅ Query classification (5 types: fiqh, aqidah, tafsir, hadith, general)
  - ✅ Madhab-aware fiqh routing
  - ✅ Authenticity hierarchy (Quran 1.0 → Seerah 0.60)
- ✅ **Phase 6 Complete**: LangGraph Server deployment
  - ✅ LangGraph Server configuration (`langgraph.json`)
  - ✅ Server entry point (195 lines)
  - ✅ Admin API endpoints (689 lines total)
  - ✅ Health checks for all services
  - ✅ Thread-based state management
  - ✅ Multi-backend support
  - ✅ Comprehensive test script (176 lines, 7/7 tests passing)
- ✅ Reranker service using LlamaIndex `LLMRerank` with multi-backend support
- ✅ All services follow LlamaIndex best practices
- ✅ Configuration system supports all backends
- ✅ **Total Stage 5 Output**: ~2,900+ lines of code across 12 files

### Next Steps
- [ ] **Phase 2**: Complete custom NodeParsers for Hadith, Tafsir, Fiqh, Seerah
- [ ] **Phase 5**: Implement multi-source retrieval with authenticity ranking (partially done in Phase 4)
- [ ] **Phase 7**: End-to-end testing & refinement with real data
- [ ] **Phase 8**: Documentation & finalization
