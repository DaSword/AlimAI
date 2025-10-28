"""
Data Migration Utility - Stage 4 Implementation

This module provides utilities to migrate existing Qdrant data to the new
universal schema without re-embedding.

Key features:
- Migrates existing Quran data to new schema
- Preserves existing embeddings (no re-computation needed)
- Updates metadata fields to match universal schema
- Adds missing fields (book_title, author, source_type, etc.)
- Supports batch processing for large collections

Usage:
    python -m backend.migrate_data --collection islamic_knowledge --source-type quran
"""

import argparse
from typing import Dict, Any, Optional
from datetime import datetime

from backend.vectordb.qdrant_manager import QdrantManager
from backend.core.models import SourceType, ChunkType, create_qdrant_payload
from backend.core.utils import setup_logging, ProgressTracker
from backend.core.config import Config

logger = setup_logging("migrate_data")


class DataMigration:
    """Handles data migration for Qdrant collections."""
    
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None):
        """
        Initialize DataMigration.
        
        Args:
            qdrant_manager: QdrantManager instance (creates default if None)
        """
        self.qdrant = qdrant_manager or QdrantManager()
        logger.info("DataMigration initialized")
    
    def migrate_quran_point(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a single Quran point to new schema.
        
        Args:
            point: Point dictionary from Qdrant
            
        Returns:
            Updated point dictionary
        """
        point_id = point.get('id')
        vector = point.get('vector', [])
        old_payload = point.get('payload', {})
        
        # Extract existing data
        verse_key = old_payload.get('verse_key', '')
        chapter_number = old_payload.get('chapter_number', 0)
        verse_number = old_payload.get('verse_number', 0)
        chapter_name = old_payload.get('chapter_name', '')
        chapter_name_arabic = old_payload.get('chapter_name_arabic', '')
        arabic_text = old_payload.get('arabic_text', '')
        english_text = old_payload.get('english_text', '')
        text_content = old_payload.get('text_content', '')
        chunk_type = old_payload.get('chunk_type', 'verse')
        chunk_index = old_payload.get('chunk_index', 0)
        tafsir_source = old_payload.get('tafsir_source')
        
        # Determine chunk type
        if chunk_type == 'tafsir' or tafsir_source:
            source_type = SourceType.TAFSIR
            chunk_type_enum = ChunkType.TAFSIR
            book_title = f"Tafsir {tafsir_source or 'Unknown'}"
            author = tafsir_source or "Unknown"
        else:
            source_type = SourceType.QURAN
            chunk_type_enum = ChunkType.VERSE
            book_title = "The Noble Quran"
            author = "Allah (Revealed)"
        
        # Create new payload with universal schema
        new_payload = create_qdrant_payload(
            source_type=source_type,
            book_title=book_title,
            book_title_arabic="القرآن الكريم" if source_type == SourceType.QURAN else None,
            author=author,
            text_content=text_content,
            arabic_text=arabic_text if source_type == SourceType.QURAN else None,
            english_text=english_text if source_type == SourceType.QURAN else None,
            topic_tags=[],  # Can be populated later
            # Source metadata
            surah_number=chapter_number,
            verse_number=verse_number,
            verse_key=verse_key,
            surah_name=chapter_name,
            surah_name_arabic=chapter_name_arabic,
            chunk_type=chunk_type_enum,
            chunk_index=chunk_index,
            tafsir_source=tafsir_source,
            # References
            related_verses=[verse_key] if tafsir_source else [],
        )
        
        # Preserve legacy fields for backward compatibility
        payload_dict = new_payload.dict(exclude_none=True)
        payload_dict.update({
            'source': old_payload.get('source', 'Quran'),
            'source_detail': old_payload.get('source_detail', ''),
            'chapter_name': chapter_name,
            'chapter_name_arabic': chapter_name_arabic,
            'verse_key': verse_key,
            'chapter_number': chapter_number,
            'verse_number': verse_number,
            'metadata': old_payload.get('metadata', {}),
        })
        
        # Create updated point
        updated_point = {
            'id': point_id,
            'vector': vector,
            'payload': payload_dict,
        }
        
        return updated_point
    
    def migrate_collection(
        self,
        collection_name: Optional[str] = None,
        source_type: SourceType = SourceType.QURAN,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Migrate an entire collection to new schema.
        
        Args:
            collection_name: Collection name (uses default if None)
            source_type: Type of source being migrated
            batch_size: Number of points to process per batch
            dry_run: If True, only simulate migration without updating
            
        Returns:
            Dictionary with migration statistics
        """
        coll_name = collection_name or self.qdrant.collection_name
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migrating collection: {coll_name}")
        logger.info(f"Source type: {source_type.value}")
        
        # Check collection exists
        if not self.qdrant.collection_exists(coll_name):
            raise ValueError(f"Collection not found: {coll_name}")
        
        # Get collection stats
        total_points = self.qdrant.count_points(coll_name)
        logger.info(f"Total points to migrate: {total_points}")
        
        if total_points == 0:
            logger.warning("Collection is empty, nothing to migrate")
            return {
                "collection_name": coll_name,
                "total_points": 0,
                "migrated_points": 0,
                "failed_points": 0,
            }
        
        # Process points in batches
        migrated_count = 0
        failed_count = 0
        offset = None
        
        tracker = ProgressTracker(total_points, "Migrating points")
        
        while True:
            # Fetch batch
            result = self.qdrant.inspect_points(
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vector=True,
            )
            
            points = result.get('result', {}).get('points', [])
            if not points:
                break
            
            # Migrate each point
            updated_points = []
            for point in points:
                try:
                    if source_type == SourceType.QURAN:
                        updated_point = self.migrate_quran_point(point)
                    else:
                        # Add other migration functions as needed
                        logger.warning(f"Migration for {source_type.value} not implemented yet")
                        updated_point = point
                    
                    updated_points.append(updated_point)
                    migrated_count += 1
                except Exception as e:
                    logger.error(f"Failed to migrate point {point.get('id')}: {e}")
                    failed_count += 1
            
            # Update points in Qdrant (unless dry run)
            if not dry_run and updated_points:
                try:
                    self.qdrant.upsert_points(updated_points)
                except Exception as e:
                    logger.error(f"Failed to upsert batch: {e}")
                    failed_count += len(updated_points)
                    migrated_count -= len(updated_points)
            
            tracker.update(len(points))
            
            # Get next offset
            next_offset = result.get('result', {}).get('next_page_offset')
            if not next_offset or next_offset == offset:
                break
            offset = next_offset
        
        tracker.finish()
        
        stats = {
            "collection_name": coll_name,
            "source_type": source_type.value,
            "total_points": total_points,
            "migrated_points": migrated_count,
            "failed_points": failed_count,
            "success_rate": (migrated_count / total_points * 100) if total_points > 0 else 0,
            "dry_run": dry_run,
        }
        
        logger.info(f"\nMigration {'simulation ' if dry_run else ''}complete:")
        logger.info(f"  - Total points: {total_points}")
        logger.info(f"  - Migrated: {migrated_count}")
        logger.info(f"  - Failed: {failed_count}")
        logger.info(f"  - Success rate: {stats['success_rate']:.2f}%")
        
        return stats
    
    def backup_collection(
        self,
        collection_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a backup of a collection before migration.
        
        Args:
            collection_name: Collection name (uses default if None)
            output_dir: Output directory for backup (uses DATA_DIR if None)
            
        Returns:
            Dictionary with backup information
        """
        coll_name = collection_name or self.qdrant.collection_name
        
        if output_dir is None:
            output_dir = str(Config.DATA_DIR)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{output_dir}/{coll_name}_backup_{timestamp}.json"
        
        logger.info(f"Creating backup: {backup_file}")
        
        stats = self.qdrant.export_collection(
            output_file=backup_file,
            with_vectors=True,
        )
        
        logger.info(f"Backup complete: {backup_file}")
        return stats
    
    def verify_migration(
        self,
        collection_name: Optional[str] = None,
        sample_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Verify that migration was successful by checking a sample of points.
        
        Args:
            collection_name: Collection name (uses default if None)
            sample_size: Number of points to sample
            
        Returns:
            Dictionary with verification results
        """
        coll_name = collection_name or self.qdrant.collection_name
        
        logger.info(f"Verifying migration for collection: {coll_name}")
        
        # Get sample points
        result = self.qdrant.inspect_points(
            limit=sample_size,
            with_payload=True,
            with_vector=False,
        )
        
        points = result.get('result', {}).get('points', [])
        
        # Check for required fields in new schema
        required_fields = ['source_type', 'book_title', 'author', 'text_content']
        
        valid_count = 0
        invalid_points = []
        
        for point in points:
            payload = point.get('payload', {})
            
            # Check if all required fields are present
            has_all_fields = all(field in payload for field in required_fields)
            
            if has_all_fields:
                valid_count += 1
            else:
                missing_fields = [f for f in required_fields if f not in payload]
                invalid_points.append({
                    'id': point.get('id'),
                    'missing_fields': missing_fields,
                })
        
        success_rate = (valid_count / len(points) * 100) if points else 0
        
        verification_results = {
            "collection_name": coll_name,
            "sample_size": len(points),
            "valid_points": valid_count,
            "invalid_points": len(invalid_points),
            "success_rate": success_rate,
            "invalid_point_details": invalid_points[:5],  # Show first 5
        }
        
        logger.info("\nVerification results:")
        logger.info(f"  - Sample size: {len(points)}")
        logger.info(f"  - Valid points: {valid_count}")
        logger.info(f"  - Invalid points: {len(invalid_points)}")
        logger.info(f"  - Success rate: {success_rate:.2f}%")
        
        if invalid_points:
            logger.warning(f"  - Found {len(invalid_points)} points with missing fields")
            logger.warning(f"  - Sample invalid points: {invalid_points[:3]}")
        
        return verification_results


def main():
    """CLI for data migration."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection to new universal schema"
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Collection name (uses default if not specified)'
    )
    parser.add_argument(
        '--source-type',
        type=str,
        choices=[st.value for st in SourceType],
        default='quran',
        help='Type of source being migrated'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of points to process per batch'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without updating data'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before migration'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify migration after completion'
    )
    
    args = parser.parse_args()
    
    # Initialize migration
    migration = DataMigration()
    
    # Create backup if requested
    if args.backup:
        logger.info("\n" + "="*60)
        logger.info("Creating backup...")
        logger.info("="*60)
        migration.backup_collection(collection_name=args.collection)
    
    # Run migration
    logger.info("\n" + "="*60)
    logger.info("Starting migration...")
    logger.info("="*60)
    
    source_type = SourceType(args.source_type)
    stats = migration.migrate_collection(
        collection_name=args.collection,
        source_type=source_type,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    
    # Verify if requested
    if args.verify and not args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("Verifying migration...")
        logger.info("="*60)
        migration.verify_migration(collection_name=args.collection)
    
    logger.info("\n" + "="*60)
    logger.info("Migration complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

