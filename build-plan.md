# Islamic Chatbot RAG - Build Plan

This document breaks down the full RAG pipeline implementation into actionable stages with clear acceptance criteria. Each stage includes both technical (coding) and non-technical (data acquisition) work.

**Note:** This project now uses **LlamaIndex** for indexing/retrieval and **LangGraph** for RAG orchestration. See `RAG-plan.md` for detailed implementation architecture, patterns, and code examples.

---

<!-- ## Stage 1: Environment Setup & Infrastructure  -  COMPLETED

**Goal:** Establish foundational infrastructure for local development and deployment.

### Tasks

#### Technical
- [ ] Update `docker-compose.yml` to include Ollama service
- [ ] Configure Qdrant service with persistent volumes
- [ ] Set up environment variables (`.env` file)
- [ ] Create `logs/` directory structure
- [ ] Update `requirements.txt` with all dependencies

#### Non-Technical
- [ ] Install Docker Desktop (if not already installed)
- [ ] Verify sufficient disk space (minimum 20GB recommended)

### Deliverables
- Working `docker-compose.yml` with services: Qdrant, Ollama
- `.env` file with configuration variables
- Updated `requirements.txt`

### Acceptance Criteria
- [ ] `docker-compose up` starts all services without errors
- [ ] Qdrant UI accessible at `http://localhost:6333/dashboard`
- [ ] Ollama service responds to health check at `http://localhost:11434`
- [ ] Python virtual environment activates and installs all dependencies

### Dependencies
None - this is the foundation for all other stages. -->

---

<!-- ## Stage 2: Backend Core Structure  --  COMPLETED

**Goal:** Create organized backend structure and migrate existing code.

### Tasks

#### Technical
- [ ] Create `backend/` directory structure
- [ ] Migrate `qdrant_manager.py` → `backend/qdrant_manager.py`
- [ ] Migrate `quran_chunker.py` → `backend/chunking.py`
- [ ] Create `backend/config.py` for centralized configuration
- [ ] Create `backend/models.py` with Pydantic schemas for state and data
- [ ] Create `backend/utils.py` for shared utilities
- [ ] Update all imports to use new structure (e.g., `from backend.qdrant_manager import QdrantManager`)

### Deliverables
- Organized `backend/` directory with flat architecture
- All existing functionality preserved in new structure
- Configuration centralized in `config.py`

### Acceptance Criteria
- [ ] Existing Quran data operations still work with migrated code
- [ ] No import errors when running migrated modules
- [ ] Configuration loaded from environment variables via `config.py`
- [ ] All modules have consistent import patterns

### Dependencies
- Stage 1 must be complete -->

---

<!-- ## Stage 3: Ollama Integration & Model Setup ✅ COMPLETED

**Goal:** Replace sentence-transformers with Ollama-based embedding service and set up LLM.

### Tasks

#### Technical
- [x] Create `backend/embeddings_service.py` with Ollama via LlamaIndex
  - ✅ Uses LlamaIndex `OllamaEmbedding` wrapper
  - ✅ Support batch embedding generation
  - ✅ Maintains interface compatibility with old `EmbeddingsManager`
  - ✅ Fallback to sentence-transformers if needed
- [x] Create `backend/llm_service.py` for chat completions
  - ✅ Uses LlamaIndex `Ollama` LLM wrapper
  - ✅ Support streaming responses
  - ✅ Handle context window management
  - ✅ Graceful error handling for memory constraints
- [x] Create `backend/reranker_service.py`
  - ✅ Uses LlamaIndex `LLMRerank` postprocessor
  - ✅ LLM-based reranking following LlamaIndex best practices
  - ✅ Graceful fallback when memory limited
- [x] Create `backend/llama_config.py`
  - ✅ Centralized LlamaIndex configuration
  - ✅ Helper functions for getting LLM and embeddings
  - ✅ Model availability checking utilities
- [x] Update `docker-compose.yml` with Ollama service

#### Non-Technical
- [x] Download and test Ollama models:
  - ✅ `nomic-embed-text:latest` (embedding - 768 dimensions)
  - ✅ `qwen2.5:3b` (chat - memory efficient)
  - ✅ `embeddinggemma:latest` (alternative embedding model)
  - ✅ Tested reranking with LLM (no dedicated reranker needed)

### Deliverables
- ✅ `backend/embeddings_service.py` - LlamaIndex OllamaEmbedding wrapper
- ✅ `backend/llm_service.py` - LlamaIndex Ollama LLM wrapper  
- ✅ `backend/reranker_service.py` - LlamaIndex LLMRerank implementation
- ✅ `backend/llama_config.py` - LlamaIndex global configuration
- ✅ `example_chat.py` - Demonstrates Chat Engines usage
- ✅ Models downloaded and tested

### Acceptance Criteria
- [x] `embeddings_service.py` generates embeddings via Ollama API
- [x] Embedding generation completes in reasonable time (<5s for 100 texts)
- [x] `llm_service.py` generates chat completions via Ollama
- [x] All models load successfully on `docker-compose up`
- [x] Reranker uses LlamaIndex LLMRerank following best practices
- [x] All services follow LlamaIndex patterns and conventions
- [x] Graceful fallback handling for memory constraints
- [x] Backward compatibility maintained

### Dependencies
- Stage 1 (Ollama service running) ✅
- Stage 2 (backend structure in place) ✅

### Notes
- Using `embeddinggemma:latest` instead of Qwen embedding (better performance)
- Using `qwen2.5:3b` instead of `qwen3:4b` (more memory efficient)
- Reranker uses LLM-based approach via `LLMRerank`
- All implementations follow LlamaIndex official documentation and examples -->

---

## Stage 4: Enhanced Data Schema & Ingestion Pipeline (LlamaIndex) ✅ COMPLETED

**Goal:** Design universal data schema and build LlamaIndex-based ingestion pipeline for all text types.

**See `RAG-plan.md` for detailed LlamaIndex architecture and implementation patterns.**

### Tasks

#### Technical
- [x] Install LlamaIndex dependencies:
  - ✅ `llama-index-core`
  - ✅ `llama-index-vector-stores-qdrant`
  - ✅ `llama-index-embeddings-ollama`
  - ✅ `llama-index-embeddings-huggingface` (fast local embeddings)
  - ✅ `llama-index-embeddings-openai` (for LM Studio)
  - ✅ `llama-index-llms-ollama`
  - ✅ `llama-index-llms-lmstudio`
  - ✅ `sentence-transformers` + `torch` (for HuggingFace backend)
- [x] Create `backend/llama/llama_config.py`:
  - ✅ Multi-backend embedding configuration (HuggingFace, Ollama, LM Studio)
  - ✅ Multi-backend LLM configuration (Ollama, LM Studio)
  - ✅ Configure Qdrant vector store integration
  - ✅ Set up LlamaIndex Settings with dynamic backend selection
  - ✅ Helper functions with backend parameters
  - ✅ Proper timeout handling for all backends
- [x] Update `backend/vectordb/qdrant_manager.py` with enhanced schema:
  - ✅ Core fields: `source_type`, `book_title`, `author`, `text_content`, etc.
  - ✅ Nested `source_metadata` for source-specific fields
  - ✅ `references` object for cross-references
  - ✅ LlamaIndex Document metadata structure
  - ✅ Flexible schema supporting all source types
- [x] Enhance `backend/ingestion/chunking.py` with LlamaIndex NodeParsers:
  - ✅ Quran chunker (verse + tafsir with hierarchical metadata)
  - ✅ Support for Hadith, Tafsir, Fiqh, Seerah (architecture ready)
  - ✅ Custom metadata extractors for each text type
  - ✅ Smart chunk sizing (500-2000 chars)
- [x] Create `backend/ingestion/ingestion.py` using LlamaIndex IngestionPipeline:
  - ✅ Define transformations (chunking, embedding)
  - ✅ Configure vector store integration
  - ✅ **Streaming batch processing** to prevent memory issues
  - ✅ Progress tracking with visual feedback
  - ✅ Handle Document → Node → Vector workflow
  - ✅ Support for all embedding backends
- [x] Create multi-backend configuration system:
  - ✅ `backend/core/config.py` - Centralized configuration
  - ✅ Environment-based backend selection
  - ✅ Support for HuggingFace (fast local), Ollama (API), LM Studio (OpenAI-compatible)
  - ✅ Separate embedding and LLM backend configuration
- [x] Update all services for multi-backend support:
  - ✅ `backend/llama/embeddings_service.py` - Dynamic backend loading
  - ✅ `backend/llama/llm_service.py` - Multi-backend LLM support
  - ✅ `backend/llama/reranker_service.py` - Backend-aware reranking
- [x] Create ingestion and search scripts:
  - ✅ `ingest_quran.py` - Full Quran ingestion with auto vector size detection
  - ✅ `search_quran.py` - Search functionality with backend awareness
  - ✅ Both scripts work with all configured backends
- [x] Create data migration utility:
  - ✅ `backend/ingestion/migrate_data.py`
  - ✅ Add `book_title`, `author`, `topic_tags` fields
  - ✅ Standardize `source_type` to "quran"
  - ✅ Metadata-only migration (no re-embedding)

### Deliverables
- ✅ LlamaIndex-based ingestion pipeline with streaming
- ✅ Universal data schema implemented in Qdrant
- ✅ Multi-backend embedding system (HuggingFace/Ollama/LM Studio)
- ✅ Multi-backend LLM system (Ollama/LM Studio)
- ✅ Chunking system with NodeParsers
- ✅ Migration utility for existing data
- ✅ Working ingestion and search scripts
- ✅ Comprehensive configuration management

### Acceptance Criteria
- [x] LlamaIndex successfully connects to all embedding backends
- [x] LlamaIndex successfully connects to Qdrant vector store
- [x] New schema supports all source types without sparse fields
- [x] NodeParsers preserve hierarchical structure and metadata
- [x] Ingestion pipeline processes JSON files with streaming batches
- [x] Backend switching works via environment variables
- [x] Streaming ingestion prevents memory exhaustion
- [x] Search functionality works with all backends
- [x] Timeout issues resolved for all backends
- [x] Can filter Qdrant by `source_type` and nested metadata fields
- [x] Auto-detects embedding dimensions for any model

### Key Achievements
- **Multi-Backend Flexibility**: Support for 3 embedding backends and 2 LLM backends
- **Performance Optimized**: HuggingFace backend provides fast local embeddings with true batching
- **Memory Efficient**: Streaming ingestion processes data in batches
- **Production Ready**: Proper error handling, logging, and timeout management
- **Developer Friendly**: Easy backend switching via .env configuration

### Dependencies
- Stage 2 (backend structure) ✅
- Stage 3 (Ollama embeddings service) ✅

---

## Stage 5: RAG Retrieval System (LangGraph Server)

**Goal:** Build intelligent retrieval system with Islam-first context formatting using LangGraph Server.

**See `RAG-plan.md` for detailed LangGraph workflow architecture and state machine design.**

### Tasks

#### Technical
- [ ] Install LangGraph/LangChain dependencies:
  - `langgraph`
  - `langchain-core`
  - `langchain-community`
  - `langchain-ollama`
  - `langgraph-cli` (for LangGraph Server)
- [ ] Create `backend/prompts.py` with system prompts:
  - Identity & methodology instructions
  - Response framework templates
  - Guardrails for Islamic accuracy
  - Prompt templates for LangChain
- [ ] Create `backend/rag_graph.py` - LangGraph workflow:
  - Define StateGraph for RAG pipeline
  - Nodes: query_classifier, query_expander, retriever, context_ranker, generator, citation_formatter
  - Edges: conditional routing based on question type
  - State management for conversation flow with persistence
  - Madhab-aware routing for fiqh questions
  - Compile graph with checkpointing enabled
- [ ] Create `backend/rag_nodes.py` - Individual workflow nodes:
  - Query classification node (fiqh/aqidah/tafsir/general)
  - Query expansion node (Islam-centric reformulation)
  - Retrieval node (LlamaIndex integration)
  - Context ranking node (authenticity weighting)
  - Generation node (Ollama LLM)
  - Citation formatting node
- [ ] Create `backend/context_formatter.py`:
  - Context ranking by authenticity (Quran > Sahih Hadith > Fiqh)
  - Multi-perspective assembly
  - Context templates for different question types
  - Integration with LangChain document formatting
- [ ] Create `backend/retrieval.py` - LlamaIndex query engine wrapper:
  - Multi-query expansion
  - Filtered search with Qdrant metadata
  - Hybrid ranking (similarity + authenticity weighting)
  - Return structured results for LangGraph
- [ ] Create `backend/server.py` - LangGraph Server setup:
  - Configure LangGraph Server
  - Register workflow graph
  - Define custom endpoints for admin operations
  - Configure authentication (if needed)
- [ ] Create `langgraph.json` - LangGraph Server configuration:
  - Define graph entry point
  - Configure dependencies
  - Set environment variables
  - Configure checkpointing/persistence
- [ ] Create admin API endpoints (in LangGraph Server or separate lightweight API):
  - `/api/admin/ingest` - Trigger ingestion
  - `/api/admin/collections` - Manage Qdrant collections
  - `/api/admin/models/status` - Check Ollama models
- [ ] Update docker-compose.yml:
  - Add LangGraph Server service
  - Configure networking with Ollama and Qdrant
  - Set up volumes for persistence

### Deliverables
- LangGraph Server with RAG workflow
- LlamaIndex query engine integration
- Admin endpoints for management operations
- System prompts and context formatting

### Acceptance Criteria
- [ ] LangGraph Server starts successfully
- [ ] LangGraph workflow executes full RAG pipeline
- [ ] Chat endpoint (`/runs/stream`) returns Islam-grounded responses
- [ ] Streaming responses work correctly
- [ ] Responses include specific source citations (Surah:Verse, Hadith references)
- [ ] Query "What is charity?" retrieves Islamic sources on Zakat/Sadaqah
- [ ] Fiqh questions return perspectives from multiple madhahib
- [ ] Workflow correctly routes based on question classification
- [ ] State persists across conversation turns with thread management
- [ ] Admin endpoints functional for ingestion and collection management
- [ ] Health check returns status of all services

### Dependencies
- Stage 3 (Ollama LLM service)
- Stage 4 (LlamaIndex ingestion and enhanced data schema)

---

## Stage 6: Frontend Development

**Goal:** Build React admin panel and chat interface connected to LangGraph Server.

### Tasks

#### Technical
- [ ] Initialize Vite + React app in `frontend/`
- [ ] Install dependencies:
  - `react-router-dom` (routing)
  - `@langchain/langgraph-sdk` (LangGraph client)
  - `axios` (API client for admin endpoints)
  - `tailwindcss` (styling)
- [ ] Create `frontend/src/api/client.js` API client:
  - LangGraph SDK client for chat
  - Axios client for admin endpoints
  - Handle streaming responses
  - Thread/conversation management
- [ ] Create `frontend/src/App.jsx` with routing:
  - `/` - Chat interface
  - `/admin` - Admin dashboard
- [ ] Create `frontend/src/pages/Chat.jsx`:
  - Message history display with conversation threads
  - Chat input box
  - Real-time streaming response rendering
  - Source citations for each response
  - Thread management (new conversation, continue thread)
- [ ] Create `frontend/src/pages/Admin.jsx` with tabs:
  - Ingestion tab
  - Collections tab
  - Models tab
- [ ] Create admin components:
  - `components/admin/IngestionPanel.jsx` - Upload/ingest files
  - `components/admin/CollectionManager.jsx` - Manage Qdrant collections
  - `components/admin/ModelStatus.jsx` - Check Ollama models
- [ ] Integrate with LangGraph Server API:
  - `POST /runs/stream` - Main chat endpoint with streaming
  - `POST /threads` - Create new conversation thread
  - `GET /threads/{thread_id}/history` - Get conversation history
- [ ] Configure CORS in LangGraph Server for local development

### Deliverables
- Complete React frontend with chat and admin interfaces
- LangGraph SDK integration for chat
- Admin API client
- Connected full-stack application

### Acceptance Criteria
- [ ] Chat interface sends messages via LangGraph SDK
- [ ] Streaming responses display in real-time
- [ ] Source citations render properly with references
- [ ] Conversation threads persist and can be resumed
- [ ] Admin panel can upload JSON files
- [ ] Ingestion panel shows progress and success/error messages
- [ ] Collections tab lists all Qdrant collections with stats
- [ ] Models tab shows loaded Ollama models
- [ ] No CORS errors when frontend calls LangGraph Server

### Dependencies
- Stage 5 (LangGraph Server with RAG workflow)

---

## Stage 7: Tier 1 Data Acquisition & Ingestion

**Goal:** Acquire and ingest the most essential Islamic texts (Tier 1 from plan).

### Tasks

#### Non-Technical
- [ ] **Acquire Sahih al-Bukhari** (already completed - in data/)
  - Download from sunnah.com API or similar source
  - Verify JSON format with book/chapter/hadith structure
  - Ensure Arabic + English text included
- [ ] **Acquire Sahih Muslim**
  - Download from sunnah.com API or similar source
  - Verify format consistency with Bukhari
- [ ] **Acquire Tafsir Ibn Kathir**
  - Download from tanzil.net, islamicstudies.info, or similar
  - Verify verse-by-verse commentary structure
  - Ensure Arabic + English available
- [ ] **Acquire Riyad al-Salihin**
  - Download from hadithcollection.com or similar
  - Verify chapter/hadith organization
- [ ] **Acquire Al-Sirah al-Nabawiyyah (Ibn Hisham)**
  - Download from islamicstudies.info or similar
  - Verify chronological event structure

#### Technical
- [ ] Ingest Sahih al-Bukhari via admin panel
- [ ] Ingest Sahih Muslim via admin panel
- [ ] Ingest Tafsir Ibn Kathir via admin panel
- [ ] Ingest Riyad al-Salihin via admin panel
- [ ] Ingest Ibn Hisham Seerah via admin panel
- [ ] Verify collection counts and sample queries

### Deliverables
- All Tier 1 texts downloaded and stored in `data/` directories
- All Tier 1 texts ingested into Qdrant
- Minimum 10,000+ points across all collections

### Acceptance Criteria
- [ ] All 5 Tier 1 texts successfully downloaded in JSON format
- [ ] Each text ingested without errors
- [ ] Qdrant collections show correct point counts:
  - Quran: ~6,236 verses (already done)
  - Sahih al-Bukhari: ~7,000+ hadiths
  - Sahih Muslim: ~7,000+ hadiths
  - Tafsir Ibn Kathir: ~6,000+ commentary chunks
  - Riyad al-Salihin: ~2,000+ hadiths
  - Ibn Hisham Seerah: ~500+ event chunks
- [ ] Sample queries return results from multiple source types
- [ ] Metadata fields populated correctly for each source type

### Dependencies
- Stage 4 (ingestion pipeline complete)
- Stage 6 (admin panel available for ingestion)

---

## Stage 8: Testing & Refinement

**Goal:** Test end-to-end functionality and refine response quality.

### Tasks

#### Technical
- [ ] Create test suite for core functionality:
  - Embedding generation
  - Chunking various text types
  - Ingestion pipeline
  - Retrieval accuracy
  - Chat endpoint responses
- [ ] Load test with diverse question types:
  - Aqidah questions ("What are the pillars of Iman?")
  - Fiqh questions ("How do I pray Witr?")
  - Tafsir questions ("What does verse 2:255 mean?")
  - General questions ("What is the purpose of life?")
  - Hadith lookup ("Tell me about hadith on intentions")
- [ ] Evaluate response quality:
  - Source grounding accuracy
  - Citation correctness
  - Madhab balance in fiqh responses
  - Authenticity ranking
- [ ] Refine based on results:
  - Adjust retrieval weights in `backend/retrieval.py`
  - Improve prompts in `backend/prompts.py`
  - Fine-tune context formatting
  - Optimize chunk sizes if needed

#### Non-Technical
- [ ] Manual testing with 20+ diverse questions
- [ ] Verify citations by checking against original sources
- [ ] Document known limitations and edge cases

### Deliverables
- Test suite covering core functionality
- Documented test results and refinements
- Optimized retrieval and prompt configuration

### Acceptance Criteria
- [ ] 90%+ of test questions return relevant, grounded responses
- [ ] All citations are verifiable and accurate
- [ ] Fiqh questions include perspectives from at least 2 madhahib
- [ ] Responses prioritize Quran/Sahih Hadith appropriately
- [ ] No hallucinations (claims without source backing)
- [ ] Response time acceptable (<10s per query)

### Dependencies
- Stage 7 (Tier 1 data ingested)

---

## Stage 9: Tier 2 & 3 Expansion (Future)

**Goal:** Gradually expand knowledge base with additional authentic texts.

### Tasks

#### Non-Technical - Tier 2 Texts
- [ ] Acquire Al-Hidayah (Hanafi fiqh)
- [ ] Acquire Al-Muwatta (Maliki fiqh)
- [ ] Acquire Matn al-Ghayah wa al-Taqrib (Shafi'i fiqh)
- [ ] Acquire 'Umdat al-Fiqh (Hanbali fiqh)
- [ ] Acquire Sunan Abi Dawud
- [ ] Acquire Jami' al-Tirmidhi

#### Non-Technical - Tier 3 Texts
- [ ] Acquire Al-Itqan fi Ulum al-Qur'an (Quranic sciences)
- [ ] Acquire Muqaddimat Ibn al-Salah (Hadith methodology)
- [ ] Acquire Al-Risala (Usul al-Fiqh)
- [ ] Acquire Al-Aqidah al-Wasitiyyah (Aqidah)
- [ ] Acquire Ihya Ulum al-Din (Spiritual purification)

#### Technical
- [ ] Ingest each text via admin panel
- [ ] Add madhab filters to retrieval system
- [ ] Enhance fiqh query handling for 4-madhab comparison
- [ ] Update prompts to leverage expanded knowledge base

### Deliverables
- 10+ additional authentic texts ingested
- Enhanced madhab-aware retrieval
- Coverage across all major Islamic sciences

### Acceptance Criteria
- [ ] All Tier 2 texts successfully ingested
- [ ] Fiqh questions automatically retrieve from all 4 madhahib
- [ ] Methodology questions return appropriate scholarly sources
- [ ] Total knowledge base exceeds 50,000+ points
- [ ] Response quality improves for specialized questions

### Dependencies
- Stage 8 (testing complete and system stable)

---

## Progress Tracking

### Current Stage
**Stage 5** - RAG Retrieval System (LangGraph Server)

### Completion Status
- [x] Stage 1: Environment Setup & Infrastructure ✅
- [x] Stage 2: Backend Core Structure ✅
- [x] Stage 3: Ollama Integration & Model Setup ✅
- [x] Stage 4: Enhanced Data Schema & Ingestion Pipeline ✅
- [ ] Stage 5: RAG Retrieval System (LangGraph Server)
- [ ] Stage 6: Frontend Development
- [ ] Stage 7: Tier 1 Data Acquisition & Ingestion
- [ ] Stage 8: Testing & Refinement
- [ ] Stage 9: Tier 2 & 3 Expansion (Future)

---

## Notes

### Stage Interdependencies
- Stages 1-3 are foundational infrastructure and must be completed in order
- Stage 4 depends on Stages 2-3 being complete
- Stage 5 depends on Stage 4 (data schema in place)
- Stage 6 can begin in parallel with Stage 5 (frontend + backend development)
- Stage 7 requires Stages 4-6 complete (need working ingestion + admin panel)
- Stage 8 requires Stage 7 (need data to test)
- Stage 9 is iterative and can be done incrementally

### Time Estimates (rough)
- Stage 1: 2-4 hours
- Stage 2: 4-6 hours
- Stage 3: 6-8 hours
- Stage 4: 8-12 hours
- Stage 5: 8-12 hours
- Stage 6: 12-16 hours
- Stage 7: 4-8 hours (mostly data acquisition time)
- Stage 8: 6-10 hours
- Stage 9: Ongoing (2-4 hours per text)

**Total estimated time for Stages 1-8: 50-75 hours**

### Critical Success Factors
1. **Data Quality:** Ensure all acquired texts are authentic and properly formatted
2. **Schema Design:** Get the Qdrant schema right in Stage 4 to avoid costly migrations
3. **Retrieval Quality:** Invest time in Stage 5 and Stage 8 to optimize retrieval
4. **User Experience:** Admin panel (Stage 6) should make ingestion effortless
5. **Source Grounding:** Every response must be verifiable against sources

