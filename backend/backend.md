# Backend Module Reference

This directory contains the core backend modules for the Islamic Chatbot RAG system.

## Architecture

**Flat structure** - No `__init__.py` files. All modules are at the same level for simplified imports.

## Modules

### `config.py` (3.0 KB)
Centralized configuration management.

**Usage:**
```python
from backend.config import Config

# Access configuration
print(Config.QDRANT_URL)
print(Config.OLLAMA_EMBEDDING_MODEL)

# Get paths
data_file = Config.get_data_path("quran.json")
log_file = Config.get_log_path("app.log")

# Print configuration
Config.print_config()
```

### `models.py` (8.9 KB)
Pydantic schemas for type-safe data structures.

**Usage:**
```python
from backend.models import (
    SourceType, QdrantPayload, RAGState,
    create_qdrant_payload
)

# Create a Quran payload
payload = create_qdrant_payload(
    source_type=SourceType.QURAN,
    book_title="The Noble Quran",
    author="Allah (Revealed)",
    text_content="...",
    surah_number=2,
    verse_number=255
)

# Create RAG state
state = RAGState(
    user_query="What is prayer in Islam?",
    max_sources=10
)
```

### `utils.py` (8.3 KB)
Shared utility functions.

**Usage:**
```python
from backend.utils import (
    setup_logging, generate_uuid,
    format_verse_reference, ProgressTracker
)

# Setup logging
logger = setup_logging("my_module")

# Generate deterministic UUID
point_id = generate_uuid("quran", "2:255")

# Format references
ref = format_verse_reference(2, 255, "Al-Baqarah")

# Track progress
tracker = ProgressTracker(total=100, description="Processing")
for i in range(100):
    tracker.update(1)
tracker.finish()
```

### `qdrant_manager.py` (17 KB)
Vector database operations.

**Usage:**
```python
from backend.qdrant_manager import QdrantManager

# Initialize manager
manager = QdrantManager()

# Create collection
manager.create_collection(vector_size=1024)

# Upsert points
points = [
    {
        "id": "uuid-1",
        "vector": [...],  # 1024-dim vector
        "payload": {...}
    }
]
manager.upsert_points(points)

# Search
results = manager.search_points(
    vector=[...],
    limit=10,
    score_threshold=0.7
)

# Export/Import
manager.export_collection("backup.json")
manager.import_collection("backup.json")
```

### `embeddings_service.py` (8.2 KB)
Text embedding generation.

**Usage:**
```python
from backend.embeddings_service import EmbeddingsService

# Initialize service
service = EmbeddingsService()

# Generate single embedding
embedding = service.generate_embedding("Bismillah...")

# Batch generation
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = service.generate_embeddings_batch(texts)

# Test
result = service.test_embedding()
```

### `chunking.py` (15 KB)
Semantic text chunking.

**Usage:**
```python
from backend.chunking import chunk_verse, chunk_plain_text

# Chunk a verse with tafsir
verse_data = {
    'verse_key': '2:255',
    'chapter_number': 2,
    'verse_number': 255,
    'chapter_name': 'Al-Baqarah',
    'arabic_text': '...',
    'english_text': '...',
    'tafsirs': {...}
}

chunks = chunk_verse(verse_data)
# Returns list of chunk dictionaries with metadata

# Chunk plain text
text = "Long text to be chunked..."
chunks = chunk_plain_text(text, max_chunk_size=1500)
```

## Import Pattern

All imports follow this pattern:

```python
from backend.config import Config
from backend.models import SourceType, QdrantPayload
from backend.utils import setup_logging
from backend.qdrant_manager import QdrantManager
from backend.embeddings_service import EmbeddingsService
from backend.chunking import chunk_verse
```

## Configuration

Configuration is loaded from environment variables (`.env` file):

```bash
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=islamic_knowledge

# Ollama (Stage 3+)
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-0.6B
OLLAMA_CHAT_MODEL=qwen3:4b

# Current embeddings (Stage 2)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
VECTOR_SIZE=1024

# RAG settings
MAX_SOURCES=10
SCORE_THRESHOLD=0.7
CHUNK_SIZE_MAX=1500
CHUNK_SIZE_MIN=100
```

## Testing

Run the test suite:

```bash
python test_backend.py
```

All tests should pass (7/7).

## Backward Compatibility

For code using old imports:

```python
# Old (still works)
from embeddings_manager import EmbeddingsManager

# New (recommended)
from backend.embeddings_service import EmbeddingsService
```

The alias `EmbeddingsManager = EmbeddingsService` ensures backward compatibility.

## Next Steps

Stage 3 will add:
- Ollama integration via LlamaIndex
- `llama_config.py` - LlamaIndex settings
- `llm_service.py` - Chat completion service
- Updated `embeddings_service.py` to use Ollama

## Module Dependencies

```
config.py (no dependencies)
  ├── utils.py
  ├── models.py
  ├── qdrant_manager.py
  ├── embeddings_service.py
  └── chunking.py
```

All modules depend on `config.py` for configuration.

