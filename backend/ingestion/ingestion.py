"""
LlamaIndex Ingestion Pipeline - Stage 4 Implementation

This module orchestrates the ingestion of Islamic texts using LlamaIndex's
IngestionPipeline. It handles:
- Loading data from JSON files
- Detecting source types
- Applying appropriate NodeParsers
- Generating embeddings via Ollama
- Storing in Qdrant vector database

The pipeline supports all Islamic text types:
- Quran (verses + tafsir)
- Hadith collections
- Tafsir commentaries
- Fiqh rulings
- Seerah events
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import NodeParser

from backend.core.config import Config
from backend.core.models import SourceType
from backend.ingestion.parsers import (
    QuranNodeParser,
    HadithNodeParser,
    TafsirNodeParser,
    FiqhNodeParser,
    SeerahNodeParser,
)
from backend.vectordb.qdrant_manager import QdrantManager
from backend.llama.llama_config import get_embed_model, check_ollama_connection
from backend.core.utils import setup_logging, ProgressTracker

logger = setup_logging("ingestion")


class IngestionManager:
    """
    Manages the ingestion of Islamic texts into the vector database.
    
    This class provides a unified interface for ingesting different types of
    Islamic texts using LlamaIndex's IngestionPipeline.
    """
    
    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        embed_model = None,
    ):
        """
        Initialize IngestionManager.
        
        Args:
            qdrant_manager: QdrantManager instance (creates default if None)
            embed_model: LlamaIndex embedding model (uses config default if None)
        """
        self.qdrant_manager = qdrant_manager or QdrantManager()
        self.embed_model = embed_model or get_embed_model()
        
        # Initialize parsers
        self.parsers = {
            SourceType.QURAN: QuranNodeParser(),
            SourceType.HADITH: HadithNodeParser(),
            SourceType.TAFSIR: TafsirNodeParser(),
            SourceType.FIQH: FiqhNodeParser(),
            SourceType.SEERAH: SeerahNodeParser(),
        }
        
        logger.info("IngestionManager initialized")
    
    def detect_source_type(self, data: Dict[str, Any]) -> Optional[SourceType]:
        """
        Detect the source type from JSON data structure.
        
        Args:
            data: JSON data dictionary
            
        Returns:
            SourceType enum or None if cannot detect
        """
        # Check for explicit source_type field
        if 'source_type' in data:
            try:
                return SourceType(data['source_type'])
            except ValueError:
                pass
        
        # Detect based on structure
        if 'verses' in data or ('verse_key' in data and 'arabic_text' in data):
            return SourceType.QURAN
        elif 'hadiths' in data or ('hadith_number' in data and 'narrator_chain' in data):
            return SourceType.HADITH
        elif 'tafsir' in data or ('tafsir_text' in data and 'verse_key' in data):
            return SourceType.TAFSIR
        elif 'fiqh' in data or ('madhab' in data and 'ruling' in data):
            return SourceType.FIQH
        elif 'seerah' in data or ('event_name' in data and 'year_hijri' in data):
            return SourceType.SEERAH
        
        logger.warning("Could not detect source type from data structure")
        return None
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with JSON data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading JSON file: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON file ({path.stat().st_size} bytes)")
        return data
    
    def json_to_documents(
        self,
        data: Dict[str, Any],
        source_type: SourceType,
    ) -> List[Document]:
        """
        Convert JSON data to LlamaIndex Documents.
        
        Args:
            data: JSON data dictionary
            source_type: Type of source (quran, hadith, etc.)
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if source_type == SourceType.QURAN:
            # Load Quran verses
            verses = data.get('verses', [])
            for verse in verses:
                # Create document with verse data in metadata
                doc = Document(
                    text=verse.get('text_for_embedding', ''),
                    metadata=verse,
                    id_=verse.get('verse_key', ''),
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} Quran documents")
        
        elif source_type == SourceType.HADITH:
            # Load hadiths
            hadiths = data.get('hadiths', [])
            for hadith in hadiths:
                # Build hadith text
                text = f"{hadith.get('english_text', '')} {hadith.get('arabic_text', '')}"
                doc = Document(
                    text=text,
                    metadata=hadith,
                    id_=f"{hadith.get('book_title', 'unknown')}:{hadith.get('hadith_number', 0)}",
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} Hadith documents")
        
        elif source_type == SourceType.TAFSIR:
            # Load tafsir entries
            tafsir_entries = data.get('tafsir_entries', [])
            for entry in tafsir_entries:
                doc = Document(
                    text=entry.get('tafsir_text', ''),
                    metadata=entry,
                    id_=f"{entry.get('tafsir_source', 'unknown')}:{entry.get('verse_key', '')}",
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} Tafsir documents")
        
        elif source_type == SourceType.FIQH:
            # Load fiqh rulings
            rulings = data.get('rulings', [])
            for ruling in rulings:
                doc = Document(
                    text=ruling.get('ruling_text', ''),
                    metadata=ruling,
                    id_=f"{ruling.get('madhab', 'unknown')}:{ruling.get('ruling_category', 'general')}",
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} Fiqh documents")
        
        elif source_type == SourceType.SEERAH:
            # Load seerah events
            events = data.get('events', [])
            for event in events:
                doc = Document(
                    text=event.get('event_text', ''),
                    metadata=event,
                    id_=f"event:{event.get('chronological_order', 0)}",
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} Seerah documents")
        
        return documents
    
    def create_pipeline(
        self,
        source_type: SourceType,
        collection_name: Optional[str] = None,
    ) -> IngestionPipeline:
        """
        Create an IngestionPipeline for a specific source type.
        
        Args:
            source_type: Type of source (quran, hadith, etc.)
            collection_name: Qdrant collection name (uses default if None)
            
        Returns:
            Configured IngestionPipeline
        """
        # Get appropriate parser for source type
        parser = self.parsers.get(source_type)
        if not parser:
            raise ValueError(f"No parser configured for source type: {source_type}")
        
        # Get vector store
        coll_name = collection_name or self.qdrant_manager.collection_name
        vector_store = self.qdrant_manager.get_vector_store(coll_name)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            transformations=[
                parser,
                self.embed_model,
            ],
            vector_store=vector_store,
        )
        
        logger.info(f"Created ingestion pipeline for {source_type.value}")
        return pipeline
    
    def ingest_file(
        self,
        file_path: str,
        source_type: Optional[SourceType] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Ingest a JSON file into the vector database.
        
        Args:
            file_path: Path to JSON file
            source_type: Type of source (auto-detected if None)
            collection_name: Qdrant collection name (uses default if None)
            batch_size: Number of documents to process per batch
            
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = datetime.now()
        
        # Check Ollama connection
        if not check_ollama_connection():
            raise RuntimeError("Cannot connect to Ollama service. Make sure it's running.")
        
        # Load JSON file
        data = self.load_json_file(file_path)
        
        # Detect source type if not provided
        if source_type is None:
            source_type = self.detect_source_type(data)
            if source_type is None:
                raise ValueError("Could not detect source type. Please specify explicitly.")
        
        logger.info(f"Ingesting as {source_type.value}")
        
        # Convert JSON to Documents
        documents = self.json_to_documents(data, source_type)
        
        if not documents:
            logger.warning("No documents created from JSON file")
            return {
                "source_type": source_type.value,
                "total_documents": 0,
                "total_nodes": 0,
                "elapsed_time": 0,
            }
        
        # Ensure collection exists
        coll_name = collection_name or self.qdrant_manager.collection_name
        if not self.qdrant_manager.collection_exists(coll_name):
            logger.info(f"Creating collection: {coll_name}")
            self.qdrant_manager.create_collection(collection_name=coll_name)
        
        # Create and run pipeline
        pipeline = self.create_pipeline(source_type, coll_name)
        
        logger.info(f"Starting ingestion of {len(documents)} documents...")
        
        # Process in batches with progress tracking
        total_nodes = 0
        tracker = ProgressTracker(len(documents), "Ingesting documents")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Run pipeline on batch
                nodes = pipeline.run(documents=batch)
                total_nodes += len(nodes)
                
                tracker.update(len(batch))
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        tracker.finish()
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "source_type": source_type.value,
            "file_path": file_path,
            "collection_name": coll_name,
            "total_documents": len(documents),
            "total_nodes": total_nodes,
            "elapsed_time": elapsed_time,
            "success": True,
        }
        
        logger.info(f"Ingestion complete:")
        logger.info(f"  - Documents: {len(documents)}")
        logger.info(f"  - Nodes: {total_nodes}")
        logger.info(f"  - Time: {elapsed_time:.2f}s")
        logger.info(f"  - Collection: {coll_name}")
        
        return stats
    
    def ingest_directory(
        self,
        directory_path: str,
        source_type: Optional[SourceType] = None,
        collection_name: Optional[str] = None,
        file_pattern: str = "*.json",
    ) -> List[Dict[str, Any]]:
        """
        Ingest all JSON files in a directory.
        
        Args:
            directory_path: Path to directory
            source_type: Type of source (auto-detected if None)
            collection_name: Qdrant collection name (uses default if None)
            file_pattern: Glob pattern for files (default: *.json)
            
        Returns:
            List of ingestion statistics for each file
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Find all JSON files
        json_files = list(directory.glob(file_pattern))
        logger.info(f"Found {len(json_files)} files matching pattern '{file_pattern}'")
        
        results = []
        
        for json_file in json_files:
            logger.info(f"\nProcessing file: {json_file.name}")
            try:
                stats = self.ingest_file(
                    str(json_file),
                    source_type=source_type,
                    collection_name=collection_name,
                )
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to ingest {json_file.name}: {e}")
                results.append({
                    "file_path": str(json_file),
                    "success": False,
                    "error": str(e),
                })
        
        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        total_docs = sum(r.get('total_documents', 0) for r in results if r.get('success', False))
        total_nodes = sum(r.get('total_nodes', 0) for r in results if r.get('success', False))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Directory Ingestion Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Files processed: {len(json_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(json_files) - successful}")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Total nodes: {total_nodes}")
        logger.info(f"{'='*60}\n")
        
        return results


def ingest_quran(
    quran_file: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to ingest Quran data.
    
    Args:
        quran_file: Path to quran.json (uses default if None)
        collection_name: Qdrant collection name (uses default if None)
        
    Returns:
        Ingestion statistics
    """
    if quran_file is None:
        quran_file = str(Config.DATA_DIR / "quran.json")
    
    manager = IngestionManager()
    return manager.ingest_file(
        quran_file,
        source_type=SourceType.QURAN,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    """Test ingestion with sample data."""
    import sys
    
    if len(sys.argv) > 1:
        # Ingest specific file
        file_path = sys.argv[1]
        source_type_str = sys.argv[2] if len(sys.argv) > 2 else None
        
        source_type = None
        if source_type_str:
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                print(f"Invalid source type: {source_type_str}")
                print(f"Valid types: {[st.value for st in SourceType]}")
                sys.exit(1)
        
        manager = IngestionManager()
        stats = manager.ingest_file(file_path, source_type=source_type)
        
        print("\nIngestion complete!")
        print(json.dumps(stats, indent=2))
    else:
        # Default: ingest Quran data
        print("No file specified. Ingesting default Quran data...")
        stats = ingest_quran()
        print("\nIngestion complete!")
        print(json.dumps(stats, indent=2))

