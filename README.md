# Alim AI - Islamic Knowledge Assistant

A production-ready RAG (Retrieval-Augmented Generation) system for Islamic knowledge, built with React + TypeScript frontend, FastAPI backend, LangGraph orchestration, LlamaIndex retrieval, and LM Studio for local model serving.

## 🌟 Features

### Chat Interface
- **Real-time Streaming Responses** - Token-by-token generation with typing indicators
- **Source Citations** - Every answer includes references to Quran, Hadith, Tafsir, or Fiqh sources
- **Conversation Threads** - Persistent conversation history with thread management
- **Islamic Design System** - Beautiful UI with emerald/gold color palette and geometric patterns
- **Dark/Light Themes** - Automatic theme switching with system preference support
- **Quick Actions** - One-click prompts for common questions

### Admin Panel
- **Data Ingestion** - Upload and process JSON files (Quran, Hadith, Tafsir, Fiqh, Seerah, Aqidah)
- **Collection Management** - View, clear, and delete Qdrant vector collections
- **Model Status** - Monitor health of embeddings, chat, and reranker servers
- **Health Checks** - Real-time status of all backend services

### RAG System
- **Intelligent Query Classification** - Automatically routes questions to appropriate sources
- **Madhab-Aware Retrieval** - Fiqh questions fetch perspectives from all 4 madhahib
- **Authenticity Ranking** - Prioritizes Quran > Sahih Hadith > other sources
- **Multi-Query Expansion** - Reformulates queries for better retrieval
- **Context Formatting** - Structures evidence hierarchically for LLM

## 🏗️ Architecture

### Tech Stack

**Frontend:**
- React 18.2.0 + TypeScript
- Vite 5.0.8 (build tool)
- Tailwind CSS 3.4.0 (styling)
- LangGraph SDK (chat operations)
- Axios (admin API)

**Backend:**
- Python 3.12
- FastAPI (web framework)
- LangGraph (RAG orchestration)
- LlamaIndex (indexing & retrieval)
- Qdrant (vector database)

**Models (LM Studio):**
- **Embeddings** - text-embedding-embeddinggemma-300m-qat (768 dimensions)
- **Chat** - qwen3-8b or similar (8.2B params, ~25 tokens/sec)
- **Reranker** - Integrated with LLM (LLM-based reranking)

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TS)                    │
│  - Chat Interface  - Admin Panel  - Theme System             │
└─────────────────┬───────────────────────────┬───────────────┘
                  │                           │
                  │ LangGraph SDK             │ Axios
                  │                           │
┌─────────────────▼───────────────────────────▼───────────────┐
│                 Backend (FastAPI + LangGraph)                │
│  - RAG Workflow  - Admin Endpoints  - CORS                   │
└───────┬──────────────────┬─────────────────┬────────────────┘
        │                  │                 │
        │ LlamaIndex       │ Qdrant Client   │ OpenAI API
        │                  │                 │
┌───────▼──────────┐ ┌─────▼──────────┐ ┌───▼────────────────┐
│   Qdrant         │ │   LM Studio    │ │   LM Studio        │
│  (Vector DB)     │ │  Embeddings    │ │  Chat + Reranker   │
│  Port: 6333      │ │  Port: 1234    │ │  Port: 1234        │
└──────────────────┘ └────────────────┘ └────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- LM Studio (download from https://lmstudio.ai)
- Docker (for Qdrant)

### 1. Clone & Setup

```bash
git clone <repository-url>
cd alimai

# Create Python virtual environment
python -m venv alimenv
source alimenv/bin/activate  # On Windows: alimenv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Start Services

```bash
# Start Qdrant (vector database)
docker-compose up -d qdrant

# Start LM Studio
# 1. Open LM Studio application
# 2. Load your embedding model (e.g., text-embedding-embeddinggemma-300m-qat)
# 3. Load your chat model (e.g., qwen3-8b)
# 4. Start the local server (default port: 1234)
# 5. Verify server is running at http://localhost:1234/v1/models
```

### 3. Start Backend API

```bash
# From project root
python -m uvicorn backend.api.webapp:app --host 0.0.0.0 --port 8123 --reload
```

### 4. Start Frontend

```bash
cd frontend
npm run dev
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8123
- API Docs: http://localhost:8123/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

## 📚 Documentation

- **[Build Plan](docs/build-plan.md)** - Complete implementation roadmap (Stages 1-10)
- **[Frontend Guide](docs/frontend.md)** - React component architecture and theming
- **[RAG Plan](docs/RAG-plan.md)** - LangGraph workflow and LlamaIndex integration
- **[Pipeline Overview](docs/islam-pipeline.md)** - Data ingestion and multi-source retrieval

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Backend Configuration
EMBEDDING_BACKEND=lmstudio  # or: huggingface, ollama
EMBEDDING_MODEL=text-embedding-embeddinggemma-300m-qat
LLM_BACKEND=lmstudio  # or: ollama
LLM_MODEL=qwen3-8b

# Service URLs
QDRANT_URL=http://localhost:6333
LMSTUDIO_URL=http://localhost:1234

# Collection Settings
COLLECTION_NAME=islamic_knowledge
VECTOR_SIZE=768
```

### Frontend Configuration

Frontend environment variables (`.env` in `frontend/`):

```bash
VITE_LANGGRAPH_URL=http://localhost:8123
```

## 📖 Usage

### Ingesting Data

1. Navigate to Admin Panel (gear icon in sidebar)
2. Go to "Ingestion" tab
3. Select source type (Quran, Hadith, Tafsir, etc.)
4. Upload JSON file
5. Monitor progress

### Asking Questions

Simply type your question in the chat interface. The system automatically:
1. Classifies query type (fiqh, aqidah, tafsir, hadith, general)
2. Expands query for better retrieval
3. Retrieves relevant sources from vector database
4. Ranks results by authenticity
5. Generates response with citations

### Managing LM Studio

**Starting LM Studio:**
1. Open LM Studio application
2. Go to "Local Server" tab
3. Select and load your models:
   - Embedding model (for vector generation)
   - Chat model (for conversations)
4. Click "Start Server" (default port: 1234)

**Testing the Server:**
```bash
# Check if server is running
curl http://localhost:1234/v1/models

# Test embeddings
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-embeddinggemma-300m-qat"}'

# Test chat
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Stopping:**
- Click "Stop Server" in LM Studio or close the application

## 🎨 Design Philosophy

### Islamic-Inspired UI
- **Emerald Green** - Represents paradise/Jannah
- **Gold** - Divine light and illumination
- **Teal** - Wisdom and tranquility
- **Geometric Patterns** - Islamic art motifs
- **Arabic Typography** - Support for RTL text and Arabic fonts

### RAG Architecture Principles
1. **Source Grounding** - Every response backed by authentic texts
2. **Authenticity Hierarchy** - Quran (1.0) > Sahih Hadith (0.85) > Tafsir (0.70) > Fiqh (0.65)
3. **Madhab Balance** - Fiqh questions include all 4 schools of thought
4. **Cross-Referencing** - Related verses/hadiths included for context
5. **No Hallucinations** - LLM synthesizes only from retrieved sources

## 📊 Project Status

### Completed Stages ✅
- **Stage 1:** Environment Setup & Infrastructure
- **Stage 2:** Backend Core Structure  
- **Stage 3:** llama.cpp Integration & Model Setup
- **Stage 4:** Enhanced Data Schema & Ingestion Pipeline
- **Stage 5:** RAG Retrieval System (LangGraph Server)
- **Stage 6:** Frontend Development

### Current Stage 🚧
- **Stage 7:** Tier 1 Data Acquisition & Ingestion

### Upcoming Stages 📅
- **Stage 8:** Testing & Refinement
- **Stage 9:** Tier 2 & 3 Expansion
- **Stage 10:** Knowledge Graph Enhancement (Optional)

## 🔬 Technical Highlights

### Multi-Backend Support
- **LM Studio** - OpenAI-compatible local server with GUI (current default)
- **HuggingFace** - Fast local embeddings with GPU acceleration
- **Ollama** - Alternative local model serving
- Easy switching via environment variables

### Performance
- **Chat Generation:** ~25 tokens/second
- **Embeddings:** ~0.2 seconds per request
- **Total RAM:** ~4-6 GB (LM Studio with loaded models)
- **Cross-Platform:** Works on Windows, macOS, and Linux

### Code Statistics
- **Backend:** ~10,000+ lines (Python)
- **Frontend:** ~2,800+ lines (TypeScript/TSX)
- **Documentation:** ~5,000+ lines (Markdown)
- **Total:** ~18,000+ lines of code

## 🤝 Contributing

This is a personal project, but feedback and suggestions are welcome. Please open an issue or reach out directly.

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **LlamaIndex** - RAG framework and integrations
- **LangGraph** - Workflow orchestration
- **Qdrant** - Vector database
- **LM Studio** - Local model serving with GUI
- **Qwen** - Chat and reranker models
- **Google** - Embedding models (embeddinggemma)

## 📞 Support

For questions or issues:
1. Check the [documentation](docs/)
2. Review the [build plan](docs/build-plan.md)
3. Open an issue on GitHub

---

**Built with ❤️ for the Muslim community**

*"Read in the name of your Lord who created" - Quran 96:1*

