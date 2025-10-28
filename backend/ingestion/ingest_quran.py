"""
Ingest Full Quran Data - Simple Script

This script ingests the complete quran.json file with the configured embedding backend.
It will delete the old collection and create a fresh one with the configured embeddings.

Usage:
    python ingest_quran.py
    
Time estimates:
    - HuggingFace backend (fast): ~1-3 hours
    - Ollama backend (slower): ~8-15 hours
"""

from pathlib import Path
from backend.core.config import Config, config
from backend.llama.llama_config import configure_llama_index, check_ollama_connection, get_embed_model
from backend.vectordb.qdrant_manager import QdrantManager
from backend.ingestion.ingestion import IngestionManager
from backend.core.models import SourceType
from backend.core.utils import setup_logging

logger = setup_logging("ingest_quran")


def main():
    """Ingest full Quran data."""
    
    logger.info("="*80)
    logger.info("FULL QURAN INGESTION")
    logger.info("="*80)
    
    # Show configuration
    logger.info(f"\nEmbedding Backend: {config.EMBEDDING_BACKEND}")
    if config.EMBEDDING_BACKEND == "huggingface":
        logger.info(f"Embedding Model: {config.EMBEDDING_MODEL}")
    else:
        logger.info(f"Embedding Model: {config.OLLAMA_EMBEDDING_MODEL}")
    
    # Check services based on backend
    logger.info("\n1. Checking services...")
    
    # Only check Ollama if using Ollama backend (for LLM we always need it)
    if not check_ollama_connection():
        logger.warning("⚠️  Cannot connect to Ollama")
        if config.EMBEDDING_BACKEND == "ollama":
            logger.error("✗ Ollama is required for embedding backend 'ollama'")
            logger.error("  docker-compose up -d ollama")
            return False
        else:
            logger.info("  (Ollama not required for HuggingFace embeddings, continuing...)")
    else:
        logger.info("✓ Ollama connected")
    
    # Configure LlamaIndex
    logger.info("\n2. Configuring LlamaIndex...")
    configure_llama_index()
    logger.info("✓ LlamaIndex configured successfully")
    
    # Auto-detect vector size from embedding model
    logger.info("\n3. Detecting vector dimensions...")
    embed_model = get_embed_model()
    test_embedding = embed_model.get_text_embedding("test")
    vector_size = len(test_embedding)
    logger.info(f"✓ Vector size: {vector_size} dimensions")
    
    # Setup Qdrant
    collection_name = f"islamic_knowledge_{config.EMBEDDING_BACKEND}"
    qdrant = QdrantManager(collection_name=collection_name)
    
    # Check if collection exists
    if qdrant.collection_exists(collection_name):
        response = input(f"\n⚠️  Collection '{collection_name}' already exists. Delete and recreate? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            logger.info(f"Deleting existing collection: {collection_name}")
            qdrant.clear_collection(delete_collection=True)
        else:
            logger.info("Aborted by user")
            return False
    
    # Create new collection with auto-detected vector size
    logger.info(f"\n4. Creating collection: {collection_name}")
    qdrant.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,  # Auto-detected from model
    )
    logger.info(f"✓ Collection created with {vector_size} dimensions")
    
    # Get Quran file path
    quran_file = Config.DATA_DIR / "quran.json"
    if not quran_file.exists():
        logger.error(f"✗ Quran file not found: {quran_file}")
        return False
    
    logger.info(f"\n5. Quran file: {quran_file}")
    logger.info(f"   Size: {quran_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Initialize ingestion manager
    manager = IngestionManager(qdrant_manager=qdrant)
    
    # Estimate time based on backend
    logger.info("\n" + "="*80)
    logger.info("⚠️  TIME ESTIMATE")
    logger.info("="*80)
    logger.info("Full Quran ingestion details:")
    logger.info("  - ~6,236 verses with tafsir")
    logger.info("  - Each verse creates ~10-20 chunks")
    logger.info("  - Total: ~50,000+ embeddings to generate")
    logger.info("")
    
    if config.EMBEDDING_BACKEND == "huggingface":
        logger.info(f"Backend: HuggingFace ({config.EMBEDDING_MODEL})")
        logger.info("  - CPU: ~1-3 hours ⚡")
        logger.info("  - GPU: ~20-40 minutes ⚡⚡⚡")
        logger.info("  (5-10x faster than Ollama!)")
    else:
        logger.info(f"Backend: Ollama ({config.OLLAMA_EMBEDDING_MODEL})")
        logger.info("  - CPU: ~8-15 hours 🐌")
        logger.info("  - GPU: ~2-4 hours")
        logger.info("  (Consider switching to HuggingFace for speed)")
    
    logger.info("="*80)
    
    response = input("\nDo you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Aborted by user")
        return False
    
    # Run ingestion
    logger.info("\n6. Starting ingestion...")
    logger.info("This will take some time. You can safely Ctrl+C and resume later.")
    logger.info("")
    
    try:
        stats = manager.ingest_file(
            file_path=str(quran_file),
            source_type=SourceType.QURAN,
            collection_name=collection_name,
            batch_size=100,
        )
        
        logger.info("\n" + "="*80)
        logger.info("INGESTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Source: {stats['source']}")
        logger.info(f"Documents processed: {stats['documents_processed']}")
        logger.info(f"Nodes created: {stats['nodes_created']}")
        logger.info(f"Time elapsed: {stats['time_elapsed']:.1f}s ({stats['time_elapsed']/3600:.1f} hours)")
        logger.info(f"Collection: {stats['collection_name']}")
        logger.info(f"Total points: {stats['total_points']}")
        logger.info("="*80)
        
        logger.info("\n✓ Quran data successfully ingested!")
        logger.info("\nYou can now:")
        logger.info("  1. Test retrieval with the new embeddings")
        logger.info("  2. Proceed to Stage 5 (RAG Retrieval System)")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Ingestion interrupted by user")
        logger.info("The collection is partially populated.")
        logger.info("You can either:")
        logger.info("  1. Continue where you left off (re-run this script)")
        logger.info("  2. Delete the collection and start fresh")
        return False
        
    except Exception as e:
        logger.error(f"\n✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # import sys
    # success = main()
    # sys.exit(0 if success else 1)

    from backend.vectordb.qdrant_manager import QdrantManager

    qdrant = QdrantManager(collection_name="islamic_knowledge_huggingface")
    stats = qdrant.get_collection_stats("islamic_knowledge_huggingface")
    print(f"Total points: {stats['points_count']}")  # Should be ~50,000+

    qdrant.search_points(vector=[0.1, 0.2, 0.3], limit=10)
