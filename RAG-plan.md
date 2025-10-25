# RAG System Implementation Plan

## Overview

Build a production-ready RAG system using **LangGraph Server** (orchestration) + **LlamaIndex** (ingestion/retrieval) + **Ollama** (embeddings/LLM). Focus: Backend only, multi-source Islamic knowledge base with intelligent routing.

---

## Phase 1: Ollama Integration & LlamaIndex Setup ✅ COMPLETED

**Goal:** Set up Ollama for embeddings, configure LlamaIndex with Qdrant integration.

### Tasks

1. **Install Dependencies** (`requirements.txt`) ✅
   ```
   llama-index-core>=0.11.0
   llama-index-vector-stores-qdrant>=0.3.0
   llama-index-embeddings-ollama>=0.3.0
   llama-index-llms-ollama>=0.3.0
   langgraph>=0.2.0
   langchain-core>=0.3.0
   langchain-ollama>=0.2.0
   ```

2. **Configure Ollama Models** (update `docker-compose.yml`) ✅

   - ✅ Pull embedding model: `nomic-embed-text:latest` (768 dimensions)
   - ✅ Pull LLM: `qwen2.5:3b` (memory efficient for local development)
   - ✅ Alternative embedding: `embeddinggemma:latest` tested and working
   - ✅ LLM-based reranking via `LLMRerank` (no dedicated reranker model needed)

3. **Create LlamaIndex Configuration** (`backend/llama_config.py`) ✅

   - ✅ Configure `OllamaEmbedding` with model name and base URL
   - ✅ Configure `Ollama` LLM with model and parameters
   - ✅ Helper functions: `get_embed_model()`, `get_llm()`
   - ✅ Connection checking: `check_ollama_connection()`, `check_model_available()`
   - ✅ Set global `Settings.embed_model` and `Settings.llm`

4. **Create Embeddings Service Wrapper** (`backend/embeddings_service.py`) ✅

   - ✅ Wrap LlamaIndex `OllamaEmbedding` for backward compatibility
   - ✅ Implement batch embedding generation
   - ✅ Tested with Islamic texts (Arabic + English)
   - ✅ Fallback to sentence-transformers if needed

5. **Create LLM Service Wrapper** (`backend/llm_service.py`) ✅

   - ✅ Wrap LlamaIndex `Ollama` LLM
   - ✅ Support streaming responses
   - ✅ Handle context window management
   - ✅ Graceful error handling for memory constraints

6. **Create Reranker Service** (`backend/reranker_service.py`) ✅
   
   - ✅ Uses LlamaIndex `LLMRerank` postprocessor
   - ✅ LLM-based reranking following official examples
   - ✅ Works with `NodeWithScore` objects
   - ✅ Graceful fallback when memory limited

### Deliverables

- ✅ Updated `requirements.txt` with LlamaIndex/LangGraph dependencies
- ✅ Updated `docker-compose.yml` with Ollama service
- ✅ `backend/llama_config.py` with LlamaIndex settings and helpers
- ✅ `backend/embeddings_service.py` - OllamaEmbedding wrapper
- ✅ `backend/llm_service.py` - Ollama LLM wrapper
- ✅ `backend/reranker_service.py` - LLMRerank implementation
- ✅ `example_chat.py` - Demonstrates Chat Engines usage

### Acceptance Criteria

- ✅ Ollama models start automatically with `docker-compose up`
- ✅ LlamaIndex successfully connects to Ollama embeddings (test with sample text)
- ✅ Embedding generation works for Arabic and English text
- ✅ Vector size matches Ollama model output (nomic-embed-text = 768 dimensions)
- ✅ LLM generates chat completions via Ollama
- ✅ Reranker follows LlamaIndex best practices (LLMRerank)
- ✅ All services handle memory constraints gracefully

### Implementation Notes

- Using `embeddinggemma:latest` - excellent for semantic search, 768 dimensions
- Using `qwen2.5:3b` - more memory efficient than larger models
- Reranker uses `LLMRerank` from `llama_index.core.postprocessor.llm_rerank`
- All implementations follow official LlamaIndex documentation
- Based on: https://developers.llamaindex.ai/python/examples/workflow/rag/

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

## Phase 3: Universal Ingestion Pipeline

**Goal:** Build LlamaIndex IngestionPipeline to process all Islamic text types.

### Tasks

1. **Create Ingestion Manager** (`backend/ingestion.py`)

   - Detect source type from JSON structure
   - Load JSON as LlamaIndex `Document` objects
   - Apply appropriate NodeParser based on source type
   - Configure transformations: Documents → Nodes → Embeddings → VectorStore
   - Support batch processing with progress tracking
   - Error handling and validation

2. **Create Document Loaders** (`backend/loaders/`)

   - `QuranLoader`: Load Quran JSON from `data/quran.json`
   - `HadithLoader`: Load Hadith JSON (Bukhari, Muslim formats)
   - `TafsirLoader`: Load Tafsir JSON
   - `FiqhLoader`: Load Fiqh JSON
   - Each returns `List[Document]` with metadata

3. **Build Ingestion CLI** (`backend/cli/ingest.py`)

   - Command-line interface for ingestion
   - Arguments: `--source-type`, `--file-path`, `--collection-name`
   - Progress bar for batch operations
   - Summary statistics (points ingested, errors, time elapsed)

4. **Re-ingest Migrated Quran Data**

   - Use new ingestion pipeline to load migrated Quran data
   - Generate embeddings with Ollama
   - Store in Qdrant with new schema
   - Verify collection stats

### Deliverables

- `backend/ingestion.py` with IngestionPipeline orchestration
- Document loaders in `backend/loaders/`
- `backend/cli/ingest.py` CLI tool
- Successfully re-ingested Quran data in new format

### Acceptance Criteria

- IngestionPipeline processes Quran data without errors
- Embeddings generated via Ollama (not sentence-transformers)
- Qdrant collection shows correct point count (~6,236 verses + tafsir chunks)
- Metadata fields populated correctly per universal schema
- Can filter by `source_type="quran"` in Qdrant

---

## Phase 4: LangGraph Workflow Design

**Goal:** Build intelligent RAG workflow with query classification, retrieval, ranking, and generation.

### Tasks

1. **Define State Schema** (`backend/models.py`)

   - Create Pydantic models for RAG state
   - `RAGState`: messages, query_type, retrieval_results, context, response, metadata
   - `QueryType`: Enum (fiqh, aqidah, tafsir, hadith, general)
   - `RetrievalResult`: text, score, source_type, metadata

2. **Create System Prompts** (`backend/prompts.py`)

   - Identity prompt: "Islamic knowledge assistant grounded in authentic sources"
   - Query classification prompt: Identify question type
   - Query expansion prompt: Reformulate to be Islam-centric
   - Generation prompt: Synthesize from sources with citations
   - Templates for each query type (fiqh, aqidah, tafsir, etc.)

3. **Implement Workflow Nodes** (`backend/rag_nodes.py`)

   - `classify_query_node`: Use LLM to classify question type (fiqh/aqidah/tafsir/general)
   - `expand_query_node`: Reformulate query for better retrieval
   - `retrieve_node`: Call LlamaIndex query engine with filters
   - `rank_context_node`: Rerank by authenticity (Quran > Sahih Hadith > Fiqh)
   - `generate_response_node`: LLM generates answer from context
   - `format_citations_node`: Structure source references

4. **Build RAG Graph** (`backend/rag_graph.py`)

   - Create `StateGraph` with RAGState schema
   - Add nodes from `rag_nodes.py`
   - Conditional edges based on query type:
     - fiqh → retrieve from multiple madhahib
     - aqidah → prioritize Quran + Sahih Hadith
     - tafsir → retrieve verse + commentary
     - general → broad search
   - Compile graph with checkpointer for state persistence

5. **Create Context Formatter** (`backend/context_formatter.py`)

   - Rank sources by authenticity
   - Structure context for LLM prompt
   - Templates for different query types
   - Include cross-references (related verses, hadiths)

### Deliverables

- `backend/models.py` with Pydantic state schemas
- `backend/prompts.py` with all system prompts
- `backend/rag_nodes.py` with workflow node implementations
- `backend/rag_graph.py` with compiled StateGraph
- `backend/context_formatter.py` with context structuring logic

### Acceptance Criteria

- RAG workflow executes from query → response
- Query classification correctly identifies question types
- Conditional routing works based on query type
- Retrieval node successfully queries LlamaIndex
- Generated responses include source citations
- State persists across conversation turns

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

## Phase 6: LangGraph Server Setup

**Goal:** Deploy LangGraph Server with RAG workflow, admin endpoints, and streaming support.

### Tasks

1. **Create LangGraph Server Configuration** (`langgraph.json`)
   ```json
   {
     "dependencies": ["backend"],
     "graphs": {
       "rag_assistant": "backend/rag_graph.py:graph"
     },
     "env": ".env"
   }
   ```

2. **Create Server Entry Point** (`backend/server.py`)

   - Initialize LangGraph Server
   - Register RAG workflow graph
   - Configure checkpointer (SQLite or Postgres for state persistence)
   - Set up CORS for local development

3. **Build Admin Endpoints** (`backend/admin/`)

   - `ingestion_api.py`: 
     - `POST /admin/ingest` - Trigger ingestion for file
     - `GET /admin/ingest/status` - Check ingestion progress
   - `collections_api.py`:
     - `GET /admin/collections` - List Qdrant collections
     - `GET /admin/collections/{name}/stats` - Collection statistics
     - `DELETE /admin/collections/{name}` - Clear collection
   - `models_api.py`:
     - `GET /admin/models/status` - Check Ollama models
     - `GET /admin/models/health` - Health check for all services

4. **Update Docker Compose** (`docker-compose.yml`)

   - Add LangGraph Server service
   - Environment variables: `QDRANT_URL`, `OLLAMA_URL`
   - Networking: link to Ollama and Qdrant
   - Volumes for state persistence
   - Health checks for all services

5. **Create Environment Config** (`.env.example`)
   ```
   QDRANT_URL=http://localhost:6333
   OLLAMA_URL=http://localhost:11434
   EMBEDDING_MODEL=nomic-embed-text
   LLM_MODEL=qwen2.5:7b
   COLLECTION_NAME=islamic_knowledge
   ```


### Deliverables

- `langgraph.json` configuration file
- `backend/server.py` with LangGraph Server setup
- Admin API endpoints in `backend/admin/`
- Updated `docker-compose.yml` with LangGraph Server
- `.env.example` with configuration

### Acceptance Criteria

- `docker-compose up` starts all services (Qdrant, Ollama, LangGraph Server)
- LangGraph Server accessible at `http://localhost:8123`
- Main chat endpoint: `POST /runs/stream` returns streaming responses
- Admin endpoints functional for ingestion and management
- Health check returns status of all connected services
- State persists across conversation turns (thread management)

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

1. **Phase 1** - Ollama + LlamaIndex Setup (required for ingestion)
2. **Phase 2** - NodeParsers (required for ingestion)
3. **Phase 3** - Ingestion Pipeline (get data into system)
4. **Phase 4** - LangGraph Workflow (RAG logic)
5. **Phase 5** - Multi-Source Retrieval (intelligent querying)
6. **Phase 6** - LangGraph Server (deployment)
7. **Phase 7** - Testing & Refinement (quality assurance)
8. **Phase 8** - Documentation (finalization)

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
- ✅ **Phase 1 Complete**: Ollama integration with LlamaIndex
- ✅ Ollama embeddings working via `embeddinggemma:latest`
- ✅ LLM service operational via `qwen2.5:3b`
- ✅ Reranker service using LlamaIndex `LLMRerank`
- ✅ All services follow LlamaIndex best practices
- ✅ All services start successfully with `docker-compose up`
