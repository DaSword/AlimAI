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

<!-- ## Stage 3: llama.cpp Integration & Model Setup ✅ COMPLETED

**Goal:** Set up native llama.cpp servers for embeddings, chat, and reranking with local GGUF models.

### Tasks

#### Technical
- [x] Download GGUF models to `models/` directory:
  - ✅ `embeddinggemma-300m-qat-Q4_K_M.gguf` (302M params, 225 MB)
  - ✅ `Qwen3-8B-Q4_K_M.gguf` (8.2B params, 4.7 GB)
  - ✅ `Qwen3-Reranker-0.6B-Q4_K_M.gguf` (595M params, 378 MB)
- [x] Create `llamacpp-manager.sh` script for server management:
  - ✅ Start/stop/restart commands for all servers
  - ✅ Health checks and status monitoring
  - ✅ Testing endpoints (embeddings, chat, reranker)
  - ✅ Streaming chat support
  - ✅ Log management
- [x] Set up three llama.cpp servers:
  - ✅ Embeddings server (port 8001) - 768 dimensions
  - ✅ Chat server (port 8002) - OpenAI-compatible with Jinja templates
  - ✅ Reranker server (port 8003) - search result reranking
- [x] Create `backend/llama/llama_config.py`:
  - ✅ Multi-backend configuration (HuggingFace, llama.cpp, LM Studio)
  - ✅ Helper functions for getting LLM and embeddings
  - ✅ Connection checking utilities
  - ✅ Proper timeout handling
- [x] Create backend service wrappers:
  - ✅ `backend/llama/embeddings_service.py` - Multi-backend support
  - ✅ `backend/llama/llm_service.py` - Multi-backend LLM with streaming
  - ✅ `backend/llama/reranker_service.py` - LLMRerank implementation

#### Non-Technical
- [x] Model downloads and verification:
  - ✅ All GGUF models downloaded and tested
  - ✅ Q4_K_M quantization for optimal speed/quality balance
  - ✅ Native ARM64 execution on Apple Silicon
  - ✅ Performance: ~25 tokens/sec (chat), ~0.2s per request (embeddings)

### Deliverables
- ✅ `llamacpp-manager.sh` - Complete server management script (486 lines)
- ✅ `docs/LLAMACPP_MANAGER.md` - Comprehensive usage documentation (247 lines)
- ✅ Three running llama.cpp servers with OpenAI-compatible APIs
- ✅ `backend/llama/llama_config.py` - Multi-backend configuration (515 lines)
- ✅ `backend/llama/embeddings_service.py` - Multi-backend embeddings wrapper
- ✅ `backend/llama/llm_service.py` - Multi-backend LLM wrapper with streaming
- ✅ `backend/llama/reranker_service.py` - LlamaIndex LLMRerank implementation
- ✅ `models/` directory with all GGUF files (~5.3 GB total)

### Acceptance Criteria
- [x] All three llama.cpp servers start successfully
- [x] Embeddings server generates 768-dimensional vectors
- [x] Chat server responds with streaming support
- [x] Reranker server ranks search results
- [x] Manager script handles all operations (start/stop/status/test)
- [x] Health checks verify all services are operational
- [x] Models run natively on Apple Silicon (ARM64)
- [x] OpenAI-compatible API endpoints work correctly
- [x] Backend services support multiple backends (HuggingFace/llama.cpp/LM Studio)
- [x] Proper timeout handling prevents hanging requests
- [x] All services follow LlamaIndex patterns

### Dependencies
- Stage 1 (Infrastructure) ✅
- Stage 2 (Backend structure in place) ✅

### Notes
- **Architecture Change:** Migrated from Ollama to native llama.cpp servers for better performance
- Using Q4_K_M quantized GGUF models for optimal memory/quality trade-off
- Native ARM64 execution provides ~25 tokens/sec for chat
- Total RAM usage: ~6-7 GB for all three servers
- OpenAI-compatible APIs enable easy integration with LlamaIndex
- Manager script provides production-ready service management
- Chat model supports special commands: `/no_think` (fast), `/think` (reasoning)
- All implementations follow LlamaIndex official documentation -->

---

<!-- ## Stage 4: Enhanced Data Schema & Ingestion Pipeline (LlamaIndex) ✅ COMPLETED

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
- Stage 3 (Ollama embeddings service) ✅ -->

---

## Stage 5: RAG Retrieval System (LangGraph Server) ✅ COMPLETED

**Goal:** Build intelligent retrieval system with Islam-first context formatting using LangGraph Server.

**See `RAG-plan.md` for detailed LangGraph workflow architecture and state machine design.**

**See `STAGE5_COMPLETE.md` for comprehensive completion report.**

### Tasks

#### Technical
- [x] Install LangGraph/LangChain dependencies:
  - ✅ `langgraph`
  - ✅ `langchain-core`
  - ✅ `langchain-community`
  - ✅ `langchain-ollama`
  - ✅ `langgraph-cli` (for LangGraph Server)
- [x] Create `backend/rag/prompts.py` with system prompts (467 lines):
  - ✅ Identity & methodology instructions
  - ✅ Response framework templates
  - ✅ Guardrails for Islamic accuracy
  - ✅ Prompt templates for LangChain
- [x] Create `backend/rag/rag_graph.py` - LangGraph workflow (300 lines):
  - ✅ Define StateGraph for RAG pipeline
  - ✅ Nodes: classify_query, expand_query, retrieve, rank_context, generate_response, format_citations
  - ✅ Edges: conditional routing based on question type
  - ✅ State management for conversation flow with persistence
  - ✅ Madhab-aware routing for fiqh questions
  - ✅ Compile graph with LangGraph Server-compatible checkpointing
- [x] Create `backend/rag/rag_nodes.py` - Individual workflow nodes (410 lines):
  - ✅ Query classification node (fiqh/aqidah/tafsir/general)
  - ✅ Query expansion node (Islam-centric reformulation)
  - ✅ Retrieval node (LlamaIndex integration)
  - ✅ Context ranking node (authenticity weighting)
  - ✅ Generation node (Ollama LLM)
  - ✅ Citation formatting node
- [x] Create `backend/rag/context_formatter.py` (502 lines):
  - ✅ Context ranking by authenticity (Quran > Sahih Hadith > Fiqh)
  - ✅ Multi-perspective assembly
  - ✅ Context templates for different question types
  - ✅ Integration with LangChain document formatting
- [x] Create `backend/rag/retrieval.py` - LlamaIndex query engine wrapper (566 lines):
  - ✅ Multi-query expansion
  - ✅ Filtered search with Qdrant metadata
  - ✅ Hybrid ranking (similarity + authenticity weighting)
  - ✅ Return structured results for LangGraph
- [x] Create `backend/api/server.py` - LangGraph Server setup (195 lines):
  - ✅ Configure LangGraph Server
  - ✅ Register workflow graph
  - ✅ Server entry point for LangGraph
- [x] Create `langgraph.json` - LangGraph Server configuration:
  - ✅ Define graph entry point (`backend.rag.rag_graph:graph`)
  - ✅ Configure dependencies
  - ✅ Set environment variables
  - ✅ Configure checkpointing/persistence
- [x] Create admin API endpoints:
  - ✅ `backend/api/admin/ingestion_api.py` (177 lines) - Ingestion management
  - ✅ `backend/api/admin/collection_api.py` (284 lines) - Collection management
  - ✅ `backend/api/admin/models_api.py` (228 lines) - Model status
- [x] Update docker-compose.yml:
  - ✅ Added LangGraph Server service notes
  - ✅ Documented networking with Ollama and Qdrant
  - ✅ Notes on volumes for persistence

### Deliverables
- ✅ LangGraph Server with RAG workflow
- ✅ LlamaIndex query engine integration
- ✅ Admin endpoints for management operations
- ✅ System prompts and context formatting
- ✅ Comprehensive test script (`test_rag_workflow.py` - 176 lines)
- ✅ Total: ~2,900+ lines of code across 12 files

### Acceptance Criteria
- [x] LangGraph Server starts successfully (run: `langgraph dev`)
- [x] LangGraph workflow executes full RAG pipeline (tested with mock data)
- ⏸️ Chat endpoint (`/runs/stream`) returns Islam-grounded responses (requires ingested data)
- ⏸️ Streaming responses work correctly (requires ingested data)
- ⏸️ Responses include specific source citations (requires ingested data)
- ⏸️ Query "What is charity?" retrieves Islamic sources on Zakat/Sadaqah (requires ingested data)
- ⏸️ Fiqh questions return perspectives from multiple madhahib (requires ingested data)
- ⏸️ Workflow correctly routes based on question classification (requires ingested data)
- [x] State management with checkpointing implemented
- [x] Admin endpoints functional for ingestion and collection management
- [x] Health checks implemented

### Key Implementation Notes
- **Import Style**: All imports use explicit module paths (e.g., `from backend.rag.rag_graph import graph`)
- **Minimal `__init__.py`**: All `__init__.py` files contain only comments or are empty
- **Authenticity Hierarchy**: Quran (1.0) > Sahih Hadith (0.85) > Aqidah (0.75) > Tafsir (0.70) > Usul (0.70) > Fiqh (0.65) > Seerah (0.60)
- **Multi-Backend Support**: Works with HuggingFace/Ollama/LM Studio backends
- **LangGraph Server**: Graph compiled without custom checkpointer (server provides automatic persistence)

### Dependencies
- Stage 3 (Ollama LLM service) ✅
- Stage 4 (LlamaIndex ingestion and enhanced data schema) ✅

**Date Completed:** October 28, 2025  
**Files Created:** 12  
**Files Updated:** 4

---

## Stage 6: Frontend Development ✅ COMPLETED

**Goal:** Build React admin panel and chat interface connected to backend API.

**Date Completed:** November 2025

### Tasks

#### Technical
- [x] Initialize Vite + React + TypeScript app in `frontend/`
- [x] Install dependencies:
  - ✅ `react-router-dom` (routing)
  - ✅ `@langchain/langgraph-sdk` (LangGraph client)
  - ✅ `axios` (API client for admin endpoints)
  - ✅ `tailwindcss` (styling)
  - ✅ `lucide-react` (icons)
  - ✅ `class-variance-authority` (component variants)
- [x] Create `frontend/src/api/client.ts` API client:
  - ✅ LangGraph SDK client for chat operations
  - ✅ Axios client for admin endpoints
  - ✅ Handle streaming responses
  - ✅ Thread/conversation management
  - ✅ Type-safe interfaces with TypeScript
- [x] Create `frontend/src/App.tsx` with routing:
  - ✅ `/` - Chat interface (default route)
  - ✅ Theme system initialization
  - ✅ BrowserRouter setup
- [x] Create `frontend/src/pages/Chat.tsx`:
  - ✅ Message history display with conversation threads
  - ✅ Chat input box with auto-resize
  - ✅ Real-time streaming response rendering
  - ✅ Source citations for each response
  - ✅ Thread management (new conversation, continue thread)
  - ✅ Sidebar with conversation list
  - ✅ Settings and admin modals
  - ✅ Islamic-themed design with decorative elements
- [x] Create admin components:
  - ✅ `components/admin/IngestionPanel.tsx` - Upload/ingest files with progress tracking
  - ✅ `components/admin/CollectionManager.tsx` - Manage Qdrant collections (list, clear, delete)
  - ✅ `components/admin/ModelStatus.tsx` - Check llama.cpp server models and health
  - ✅ `components/chat/AdminModal.tsx` - Tabbed admin interface
  - ✅ `components/chat/SettingsModal.tsx` - User settings and theme selection
- [x] Create chat interface components:
  - ✅ `components/chat/ChatSidebar.tsx` - Collapsible sidebar with threads
  - ✅ `components/chat/ChatInput.tsx` - Auto-resizing text input
  - ✅ `components/chat/MessageBubble.tsx` - Message display with citations
  - ✅ `components/chat/WelcomeScreen.tsx` - Initial screen with quick actions
  - ✅ `components/chat/IslamicDecorations.tsx` - SVG decorative elements
- [x] Create reusable UI components:
  - ✅ `components/ui/button.tsx` - Button with variants
  - ✅ `components/ui/card.tsx` - Card container components
  - ✅ `components/ui/input.tsx` - Text input
  - ✅ `components/ui/textarea.tsx` - Multi-line input
  - ✅ `components/ui/tabs.tsx` - Tabbed interface
- [x] Implement comprehensive theming system:
  - ✅ Dark/Light/System theme modes
  - ✅ Islamic-inspired color palette (emerald, gold, teal)
  - ✅ CSS custom properties for runtime theme switching
  - ✅ Persistent theme preferences (localStorage)
  - ✅ Islamic geometric patterns and decorations
- [x] Integrate with backend API:
  - ✅ Chat endpoints via LangGraph SDK
  - ✅ Admin endpoints via Axios
  - ✅ CORS configuration in backend
  - ✅ Environment variable configuration
- [x] Configure CORS in backend for local development

### Deliverables
- ✅ Complete React + TypeScript frontend with chat and admin interfaces (~2,800+ lines of code)
- ✅ LangGraph SDK integration for chat operations
- ✅ Admin API client for management operations
- ✅ Connected full-stack application
- ✅ Islamic-themed design system with custom components
- ✅ Comprehensive theme system (dark/light/system)
- ✅ Production-ready build configuration

### Acceptance Criteria
- [x] Chat interface sends messages via LangGraph SDK
- [x] Streaming responses display in real-time
- [x] Source citations render properly with references
- [x] Conversation threads persist and can be resumed
- [x] Admin panel can upload JSON files
- [x] Ingestion panel shows progress and success/error messages
- [x] Collections tab lists all Qdrant collections with stats
- [x] Models tab shows loaded llama.cpp server models
- [x] No CORS errors when frontend calls backend
- [x] Theme switching works seamlessly
- [x] Responsive design works on different screen sizes
- [x] TypeScript provides full type safety
- [x] Build process creates optimized production bundle

### Key Features Implemented

**Chat Interface:**
- Real-time streaming responses with typing indicators
- Source citations with book, reference, and text excerpts
- Conversation thread management
- Auto-scrolling to latest messages
- Islamic decorative elements (Bismillah, geometric patterns)
- Quick action cards for common queries

**Admin Panel:**
- File upload with source type selection (Quran, Hadith, Tafsir, Fiqh, Seerah, Aqidah)
- Collection management (list, clear, delete, view stats)
- Model status monitoring (embeddings, chat, reranker servers)
- Health checks for all backend services
- Progress tracking for ingestion operations

**Design System:**
- Islamic color palette (emerald green, gold, teal)
- Custom UI components with variants
- Dark/light theme support
- Arabic typography support (Amiri, Scheherazade New fonts)
- Geometric patterns and ornamental elements
- Responsive layout with collapsible sidebar

**Technical Stack:**
- React 18.2.0 + TypeScript
- Vite 5.0.8 (build tool)
- Tailwind CSS 3.4.0 (styling)
- React Router DOM 6.21.0 (routing)
- LangGraph SDK (chat operations)
- Axios (admin API calls)
- Lucide React (icons)

### Files Created (Frontend)
- `frontend/src/App.tsx` - Root component with routing
- `frontend/src/main.tsx` - Application entry point
- `frontend/src/pages/Chat.tsx` - Main chat page (450+ lines)
- `frontend/src/api/client.ts` - API client with TypeScript types
- `frontend/src/components/chat/` - 8 chat components
- `frontend/src/components/admin/` - 3 admin components
- `frontend/src/components/ui/` - 5 reusable UI components
- `frontend/src/index.css` - Global styles and theme definitions
- `frontend/tailwind.config.js` - Tailwind configuration
- `frontend/vite.config.ts` - Vite build configuration
- `frontend/tsconfig*.json` - TypeScript configuration files
- `frontend/package.json` - Dependencies and scripts

### Architecture Improvements
- **Type Safety:** Full TypeScript implementation with interfaces for all data structures
- **Component Architecture:** Modular, reusable components with clear separation of concerns
- **State Management:** React hooks with proper state lifting and prop drilling patterns
- **API Abstraction:** Centralized API client with typed request/response interfaces
- **Theme System:** CSS custom properties enable runtime theme switching without rebuilds
- **Build Optimization:** Vite provides fast dev server and optimized production builds

### Dependencies
- Stage 5 (Backend API with admin endpoints) ✅
- llama.cpp servers running (embeddings, chat, reranker) ✅

**Total Frontend Output:** ~2,800+ lines of TypeScript/TSX across 25+ files

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

## Stage 10: Knowledge Graph Enhancement

**Goal:** Enhance the RAG system with explicit relationship tracking for richer context retrieval using Qdrant's payload storage.

### Tasks

#### Technical - Graph Schema Design

- [ ] **Extend Qdrant Schema** (`backend/vectordb/qdrant_manager.py`)
  - Add `graph` field to payload structure
  - Define relationship types (has_tafsir, narrated_by, supports_ruling, etc.)
  - Implement bidirectional linking (forward + backward edges)
  - Add edge metadata support (weights, confidence scores)

- [ ] **Create Graph Manager** (`backend/vectordb/graph_manager.py`)
  - `QdrantGraphManager` class for graph operations
  - `get_neighbors(node_id, edge_type)` - Get connected nodes
  - `get_all_tafsir_for_verse(verse_key)` - All commentary on a verse
  - `get_narrator_chain(hadith_id)` - Complete isnad chain
  - `get_madhab_rulings(topic)` - Rulings from all madhahib
  - `find_path(start_id, end_id, max_depth)` - Path finding between nodes
  - `get_scholar_lineage(scholar_id)` - Teacher-student chains
  - `traverse_depth_n(node_id, edge_types, depth)` - N-hop traversal

#### Technical - Ingestion Updates

- [ ] **Update NodeParsers** (`backend/ingestion/parsers.py`)
  - `QuranNodeParser`: Extract has_tafsir, related_to, mentions_concept relationships
  - `HadithNodeParser`: Extract narrated_by, supports_ruling, references_verse chains
  - `TafsirNodeParser`: Extract commentary_on, authored_by, cites_hadith links
  - `FiqhNodeParser`: Extract based_on_verse, based_on_hadith, madhab_view connections
  - Add relationship extraction methods to each parser
  - Support automatic bidirectional link creation

- [ ] **Create Relationship Extractors** (`backend/ingestion/relationship_extractors.py`)
  - `extract_verse_references(text)` - Find Quran citations in text
  - `extract_hadith_references(text)` - Find hadith citations
  - `extract_narrator_chain(hadith_data)` - Parse isnad into nodes
  - `extract_concepts(text)` - Identify Islamic concepts mentioned
  - `extract_scholar_references(metadata)` - Identify scholars
  - Use regex, NLP, and LLM-based extraction

- [ ] **Update Ingestion Pipeline** (`backend/ingestion/ingestion.py`)
  - Add relationship extraction phase after chunking
  - Build relationship index during ingestion
  - Validate bidirectional links (ensure backward edges exist)
  - Store graph metadata in Qdrant payload

#### Technical - Enhanced Retrieval

- [ ] **Create Graph-Enhanced Retriever** (`backend/rag/graph_retrieval.py`)
  - `retrieve_with_graph_expansion(query, depth)` - Vector search + graph traversal
  - `get_verse_with_tafsir(query)` - Verse + all commentary
  - `get_hadith_with_context(query)` - Hadith + supporting fiqh rulings
  - `get_ruling_with_evidence(query)` - Fiqh ruling + Quran/hadith sources
  - `traverse_scholar_network(scholar_name)` - Scholar and their works
  - Hybrid scoring: combine vector similarity + graph distance

- [ ] **Update RAG Nodes** (`backend/rag/rag_nodes.py`)
  - Modify `retrieve_node` to use graph expansion
  - Add `graph_expansion_node` for controlled traversal
  - Update context formatter to display graph relationships
  - Add relationship metadata to citations

- [ ] **Update RAG Graph** (`backend/rag/rag_graph.py`)
  - Add conditional edge for graph expansion
  - Route based on query type (enable graph for tafsir/fiqh, optional for others)
  - Add depth control based on query complexity

#### Technical - Admin UI Integration

- [ ] **Add Graph Visualization Endpoint** (`backend/api/admin/graph_api.py`)
  - `GET /api/admin/graph/node/{node_id}` - Get node and immediate neighbors
  - `GET /api/admin/graph/relationships` - Get relationship type counts
  - `POST /api/admin/graph/path` - Find path between two nodes
  - `GET /api/admin/graph/stats` - Graph statistics (nodes, edges, density)
  - Return graph data in format for frontend visualization

- [ ] **Update Frontend Admin Panel** (`frontend/src/components/admin/GraphViewer.tsx`)
  - Display node details and relationships
  - Interactive graph visualization (using react-force-graph or similar)
  - Relationship type filtering
  - Node search and navigation

#### Non-Technical - Data Enrichment

- [ ] **Enrich Existing Quran Data**
  - Add tafsir relationships for existing verses
  - Add sequential verse relationships (2:1 → 2:2)
  - Add thematic relationships (fasting verses)
  - Add concept tags

- [ ] **Build Narrator Database**
  - Extract unique narrators from hadith collections
  - Build narrator chains (isnad networks)
  - Add reliability grades if available

- [ ] **Build Scholar Knowledge Base**
  - Create scholar profiles (birth/death, madhab, teachers, students)
  - Link scholars to their works (tafsir, fiqh texts)
  - Build teacher-student chains

- [ ] **Create Concept Taxonomy**
  - Define Islamic concepts hierarchy (e.g., Tawhid → Divine Names)
  - Tag texts with concepts
  - Create concept-to-concept relationships

### Deliverables

- Enhanced Qdrant schema with graph relationships
- `backend/vectordb/graph_manager.py` - Graph operations (300+ lines)
- Updated parsers with relationship extraction (150+ lines per parser)
- `backend/ingestion/relationship_extractors.py` - Extraction utilities (400+ lines)
- `backend/rag/graph_retrieval.py` - Graph-enhanced retrieval (350+ lines)
- Updated RAG workflow with graph expansion
- `backend/api/admin/graph_api.py` - Graph management endpoints (200+ lines)
- Frontend graph visualization component (250+ lines)
- Enriched knowledge base with explicit relationships

### Acceptance Criteria

- [ ] Graph schema successfully stores relationships in Qdrant payload
- [ ] Parsers extract relationships during ingestion automatically
- [ ] Bidirectional links maintained (verse ↔ tafsir)
- [ ] Graph traversal operations work correctly (1-hop, 2-hop, path-finding)
- [ ] Query "What does verse 2:183 mean?" returns verse + all tafsir via graph
- [ ] Query "Hadith on fasting" returns hadith + related fiqh rulings
- [ ] Narrator chains correctly extracted and traversable
- [ ] Graph expansion improves answer quality in testing
- [ ] Admin UI displays graph relationships and statistics
- [ ] Graph visualization shows connected knowledge
- [ ] No performance degradation (graph operations < 100ms)
- [ ] Memory usage remains reasonable (relationships stored efficiently)

### Implementation Approach

**Phase 1: Schema & Manager (Week 1)**
1. Extend Qdrant schema with graph field
2. Implement QdrantGraphManager with basic operations
3. Test with mock data

**Phase 2: Ingestion (Week 2)**
1. Update parsers to extract relationships
2. Create relationship extractors
3. Test with Quran data (verse-tafsir links)

**Phase 3: Retrieval Enhancement (Week 3)**
1. Build graph-enhanced retriever
2. Update RAG workflow nodes
3. Test hybrid vector + graph retrieval

**Phase 4: Admin & Visualization (Week 4)**
1. Add graph API endpoints
2. Build frontend graph viewer
3. End-to-end testing

**Phase 5: Data Enrichment (Ongoing)**
1. Enrich existing data with relationships
2. Build narrator and scholar databases
3. Create concept taxonomy

### Use Cases Enabled

1. **Deep Tafsir Queries**: "What does verse X mean?" → Verse + all available tafsir
2. **Evidence-Based Rulings**: "Fiqh ruling on Y" → Ruling + supporting Quran/hadith
3. **Scholar Research**: "Who was Ibn Kathir?" → Biography + works + students + teachers
4. **Narrator Verification**: "Who narrated hadith X?" → Complete isnad chain
5. **Concept Exploration**: "What is Tawhid?" → Definition + verses + hadiths + scholarly works
6. **Cross-Madhab Comparison**: "Prayer ruling" → All 4 madhab views + their evidence
7. **Chronological Context**: "What was revealed after verse X?" → Revelation sequence
8. **Thematic Clustering**: "All verses about fasting" → Connected verse network

### Benefits Over Pure Vector Search

1. **Completeness**: Get ALL tafsir for a verse, not just top-k similar
2. **Precision**: Direct relationships more accurate than semantic similarity
3. **Provenance**: Trace chains of evidence (verse → hadith → fiqh ruling)
4. **Context**: Understand connections between concepts/scholars/texts
5. **Exploration**: Navigate knowledge graph interactively
6. **Authenticity**: Follow isnad chains for hadith verification

### Dependencies
- Stage 5 (RAG system complete)
- Stage 7 (Data ingested and available)

### Estimated Time
- Planning & Schema Design: 4-6 hours
- Implementation: 30-40 hours
- Data Enrichment: 10-15 hours
- Testing & Refinement: 8-10 hours

**Total: 50-70 hours**

---

## Progress Tracking

### Current Stage
**Stage 7** - Tier 1 Data Acquisition & Ingestion

### Completion Status
- [x] Stage 1: Environment Setup & Infrastructure ✅
- [x] Stage 2: Backend Core Structure ✅
- [x] Stage 3: llama.cpp Integration & Model Setup ✅
- [x] Stage 4: Enhanced Data Schema & Ingestion Pipeline ✅
- [x] Stage 5: RAG Retrieval System (LangGraph Server) ✅ **COMPLETED October 28, 2025**
- [x] Stage 6: Frontend Development ✅ **COMPLETED November 7, 2025**
- [ ] Stage 7: Tier 1 Data Acquisition & Ingestion
- [ ] Stage 8: Testing & Refinement
- [ ] Stage 9: Tier 2 & 3 Expansion (Future)
- [ ] Stage 10: Knowledge Graph Enhancement (Optional, recommended after Stage 8)

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
- Stage 10 requires Stages 5 & 7 complete (need RAG system + data to enhance)
- Stage 10 is optional but highly recommended for maximum answer quality

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
- Stage 10: 50-70 hours (knowledge graph enhancement)

**Total estimated time for Stages 1-8: 50-75 hours**
**Total with Stage 10: 100-145 hours**

### Critical Success Factors
1. **Data Quality:** Ensure all acquired texts are authentic and properly formatted
2. **Schema Design:** Get the Qdrant schema right in Stage 4 to avoid costly migrations
3. **Retrieval Quality:** Invest time in Stage 5 and Stage 8 to optimize retrieval
4. **User Experience:** Admin panel (Stage 6) should make ingestion effortless
5. **Source Grounding:** Every response must be verifiable against sources

