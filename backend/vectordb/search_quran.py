"""
Search Quran Data - Simple Query Script

This script allows you to query your ingested Quran collection.
It uses the same embedding backend that was used for ingestion.

Usage:
    python search_quran.py "What does Surah Al-Fatiha say?"
    python search_quran.py --interactive
"""

import argparse
from llama_index.core import VectorStoreIndex
from backend.core.config import config
from backend.llama.llama_config import configure_llama_index
from backend.vectordb.qdrant_manager import QdrantManager
from backend.core.utils import setup_logging

logger = setup_logging("search_quran")


def search(query: str, collection_name: str = None, top_k: int = 5):
    """
    Search the Quran collection.
    
    Args:
        query: Search query
        collection_name: Collection name (defaults to backend-specific name)
        top_k: Number of results to return
    """
    # Use backend-specific collection name if not provided
    if collection_name is None:
        collection_name = config.QDRANT_COLLECTION_NAME
    
    logger.info(f"Searching collection: {collection_name}")
    logger.info(f"Query: {query}")
    logger.info(f"Embedding Backend: {config.EMBEDDING_BACKEND}")
    
    # Configure LlamaIndex with same backend used for ingestion
    logger.info("\nConfiguring embeddings...")
    configure_llama_index()
    
    # Get Qdrant vector store
    qdrant = QdrantManager(collection_name=collection_name)
    
    # Check if collection exists
    if not qdrant.collection_exists(collection_name):
        logger.error(f"✗ Collection '{collection_name}' not found!")
        logger.error("\nAvailable collections:")
        collections = qdrant.list_collections()
        for coll in collections:
            logger.error(f"  - {coll}")
        return None
    
    # Get collection stats
    stats = qdrant.get_collection_stats(collection_name)
    logger.info(f"✓ Collection found: {stats['points_count']} points")
    
    # Create vector store and index
    logger.info("\nCreating query engine...")
    vector_store = qdrant.get_vector_store(collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="tree_summarize",
    )
    
    # Execute query
    logger.info("\nSearching...\n")
    response = query_engine.query(query)
    
    # Display results
    print("="*80)
    print("SEARCH RESULTS")
    print("="*80)
    print(f"\nQuery: {query}\n")
    print("Answer:")
    print("-"*80)
    print(response.response)
    print("-"*80)
    
    # Show sources
    if hasattr(response, 'source_nodes') and response.source_nodes:
        print(f"\n📚 Sources ({len(response.source_nodes)} results):")
        print("-"*80)
        
        for i, node in enumerate(response.source_nodes, 1):
            metadata = node.metadata
            score = node.score if hasattr(node, 'score') else 0.0
            
            print(f"\n{i}. Score: {score:.3f}")
            
            # Show metadata based on what's available
            if 'verse_key' in metadata.get('source_metadata', {}):
                verse_key = metadata['source_metadata']['verse_key']
                surah_name = metadata['source_metadata'].get('surah_name', 'Unknown')
                print(f"   Source: {surah_name} ({verse_key})")
            elif 'book_title' in metadata:
                print(f"   Source: {metadata['book_title']}")
            
            # Show chunk type
            if 'source_metadata' in metadata and 'chunk_type' in metadata['source_metadata']:
                chunk_type = metadata['source_metadata']['chunk_type']
                print(f"   Type: {chunk_type}")
            
            # Show preview of text
            text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            print(f"   Text: {text_preview}")
    
    print("\n" + "="*80)
    
    return response


def interactive_mode(collection_name: str = None):
    """Interactive search mode."""
    print("\n" + "="*80)
    print("INTERACTIVE SEARCH MODE")
    print("="*80)
    print(f"Backend: {config.EMBEDDING_BACKEND}")
    
    if collection_name is None:
        collection_name = config.QDRANT_COLLECTION_NAME
    
    print(f"Collection: {collection_name}")
    print("\nType your questions below. Type 'exit' or 'quit' to stop.")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            search(query, collection_name=collection_name)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Search the Quran collection")
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (optional if using --interactive)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    parser.add_argument(
        '--collection',
        type=str,
        help='Collection name (defaults to islamic_knowledge_<backend>)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.collection)
    elif args.query:
        search(args.query, args.collection, args.top_k)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python search_quran.py \"What is the meaning of Al-Fatiha?\"")
        print("  python search_quran.py --interactive")
        print("  python search_quran.py \"Tell me about prayer\" --top-k 10")


if __name__ == "__main__":
    main()

